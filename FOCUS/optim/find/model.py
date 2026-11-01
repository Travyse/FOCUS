
# Inlined from FIND (https://github.com/OllieBoyne/FIND, MIT licensed - see
# FIND_LICENSE.md). Modified for self-containment:
#  * imports rewritten to reference the inlined FOCUS.optim.find modules;
#  * training-only components removed (ModelWithLoss, SUPRModel and their
#    imports of the pytorch3d rasterizer-based renderer / losses / SUPR body
#    model, none of which are used by FOCUS);
#  * the unused `cfg` import removed. Model-building and inference behavior
#    is otherwise unchanged.

import os
import torch
from FOCUS.optim.find.fourier_feature_transform import FourierFeatureTransform
import numpy as np
from typing import Union

from torch.nn import functional as F
from FOCUS.optim.find.opts import Opts

from pytorch3d.structures import Meshes
from pytorch3d.transforms import euler_angles_to_matrix, Transform3d
from pytorch3d.renderer import TexturesVertex, TexturesUV
from pytorch3d.io import load_obj as p3d_load_obj
from pytorch3d.ops import sample_points_from_meshes
from scipy.io import loadmat

from FOCUS.optim.find.pytorch3d_tools import extend_template
import trimesh

nn = torch.nn


def load_obj(loc, skip_materials=True):
	resolver = None
	if not skip_materials:
		directory = os.path.split(loc)[0]
		resolver = trimesh.visual.resolvers.FilePathResolver(directory)

	with open(loc, 'rb') as f:
		mesh = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(f, skip_materials=skip_materials, resolver=resolver))
	return mesh


def _to_numpy(arr):
	if torch.is_tensor(arr):
		return arr.cpu().detach().numpy()
	return arr


def make_params_list(*params):
	"""Produce list of parameters"""
	out = []
	for param in params:
		if hasattr(param, 'parameters'):
			out += list(param.parameters())
		elif isinstance(param, nn.Parameter):
			out += [param]
		elif param is None:
			pass
		else:
			raise ValueError(f"Param type {type(param)} not understood.")

	return out


_activations = {
	'relu': nn.ReLU,
	'tanh': nn.Tanh,
	'sigmoid': nn.Sigmoid,
}


class ProgressiveEncoding(nn.Module):
	def __init__(self, mapping_size, T, d=3, apply=True, device='cuda'):
		super(ProgressiveEncoding, self).__init__()
		self._t = 0
		self.n = mapping_size
		self.T = T
		self.d = d
		self._tau = 2 * self.n / self.T
		self.indices = torch.tensor([i for i in range(self.n)], device=device)
		self.apply = apply
		self.device = device

	def forward(self, x):
		alpha = ((self._t - self._tau * self.indices) / self._tau).clamp(0, 1).repeat(
			2)  # no need to reduce d or to check cases
		if not self.apply:
			alpha = torch.ones_like(alpha, device=self.device)  ## this layer means pure ffn without progress.
		alpha = torch.cat([torch.ones(self.d, device=self.device), alpha], dim=0)
		self._t += 1
		return x * alpha


class LatentVector(nn.Module):
	"""Learned latent vector, optimized for all the data presented."""

	def __init__(self, dataset_size=None, vec_size=512, name='', device='cuda', key=None, labels: list = None,
				 init_values=None):
		"""
		:param dataset_size:
		:param vec_size:
		:param name:
		:param device:
		:param init_values: Either float or numpy array of length vec_size to initialize vector from

		:param labels: per item labels, overrides dataset_size
		:param key: the key used to sample these labels from a dataset
		"""
		super().__init__()
		self.dataset_size = dataset_size
		self.vec_size = vec_size

		if labels is not None:
			dataset_size = len(labels)
		self.labels = labels
		self.key = key

		vec_init = torch.zeros(dataset_size, vec_size)
		if init_values is not None:
			if isinstance(init_values, np.ndarray):
				init_values = torch.from_numpy(init_values).unsqueeze(0).float()
			else:
				raise NotImplementedError
			vec_init[:] = init_values

		self.data = nn.Parameter(vec_init.to(device))
		self.name = name

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, idx):
		if isinstance(idx, int):
			return self.data[idx]

		if isinstance(idx, torch.Tensor):
			return self.data[idx]

		if isinstance(idx, str):
			assert self.labels is not None, f"Tried to access item {idx} from LatentVector {self.name}, LV does not have labels"
			i = self.labels.index(idx)
			return self.data[i]

		if isinstance(idx, list):
			if isinstance(idx[0], int):
				return self.data[idx]

			if isinstance(idx[0], str):
				assert self.labels is not None, f"Tried to access item {idx} from LatentVector {self.name}, LV does not have labels"
				i = [self.labels.index(o) for o in idx]
				return self.data[i]

		else:
			raise NotImplementedError(f"Didn't understand indexing of LatentVector {self.name}, type {type(idx)}")


class Model(nn.Module):
	def save_model(self, out_dir='models/tmp', fname='model_tmp'):
		data = {'state_dict': self.state_dict()}
		data['params'] = self.params

		os.makedirs(out_dir, exist_ok=True)
		torch.save(data, os.path.join(out_dir, fname + '.pth'))

	def freeze(self):
		for param in self.parameters():
			param.requires_grad = False

	@classmethod
	def load(cls, file, device='cuda', opts=None, **kwargs):
		ext = os.path.splitext(file)[-1]
		assert ext == '.pth', f"Generic models can only load from .pth files - received `{ext}` file."

		data = torch.load(file, map_location=device, weights_only=False)

		# If loading new latents, override train_size and val_size from params
		if opts.dont_load_latents:
			data['params']['train_size'] = kwargs.get('train_size', 1)
			data['params']['val_size'] = kwargs.get('val_size', 1)
			data['params']['latent_labels'] = kwargs.get('latent_labels', None)  # Rewrite latent labels parameter

		state_dict = data['state_dict']
		model = cls(**data['params'], device=device, opts=opts)
		model.configure_template(state_dict, device=device)

		mod_state_dict = {}
		x = len(state_dict)
		if opts is not None and opts.dont_load_latents:
			for key in state_dict:
				if key not in [f + '.data' for f in
							   ['shapevec', 'shapevec_val', 'texvec', 'texvec_val', 'posevec', 'posevec_val', 'reg',
								'reg_val']]:
					mod_state_dict[key] = state_dict[key]

			state_dict = mod_state_dict

		model.load_state_dict(state_dict, strict=False)
		return model


def load_template_classes(file, device='cuda'):
	"""Load per vertex template classes from pretrained model"""
	data = torch.load(file, map_location=device)
	state_dict = data['state_dict']
	return state_dict['features']


class NeuralDisplacementField(Model):
	def __init__(self, sigma=10, depth=4, width=256, encoding='gaussian', dispdepth=3, coldepth=3, normratio=0.1,
				 clamp=None,
				 normclamp=None, niter=6000, input_dim=3, positional_encoding=True, progressive_encoding=False,
				 exclude=0,
				 use_shapevec=False, train_size=None, val_size=None, shapevec_size=256,
				 use_texvec=False, texvec_size=256, use_posevec=False, posevec_size=256,
				 template_mesh_loc=None, device='cuda',
				 restyle_features_per_vertex=False, restyle_cluster_per_vertex=False,
				 use_avg_colour=False, opts=None,
				 latent_labels: dict = None):
		"""Model predicts (x,y,z) offsets from a template mesh.
		Also stores per data point shape vector and (pose vector).
		If :param template_mesh: is given, it is used to initialize the template mesh
		"""

		super(NeuralDisplacementField, self).__init__()

		self.params = dict(sigma=sigma, depth=depth, width=width, encoding=encoding, dispdepth=dispdepth,
						   coldepth=coldepth,
						   progressive_encoding=progressive_encoding, positional_encoding=positional_encoding,
						   train_size=train_size, val_size=val_size,
						   use_shapevec=use_shapevec, shapevec_size=shapevec_size, use_texvec=use_texvec,
						   texvec_size=texvec_size, use_posevec=use_posevec, posevec_size=posevec_size,
						   restyle_features_per_vertex=restyle_features_per_vertex,
						   restyle_cluster_per_vertex=restyle_cluster_per_vertex,
						   use_avg_colour=use_avg_colour, latent_labels=latent_labels)

		self.pe = ProgressiveEncoding(mapping_size=width, T=niter, d=input_dim, device=device)
		self.clamp = clamp
		self.normclamp = normclamp
		self.normratio = normratio
		encoding_layers, layers = [], []
		if encoding == 'gaussian':
			if positional_encoding:
				encoding_layers.append(FourierFeatureTransform(input_dim, width, sigma, exclude))
				input_size = width * 2 + input_dim

			else:
				input_size = input_dim

			if progressive_encoding:
				encoding_layers.append(self.pe)

			layers.append(nn.Linear(input_size, width))
			layers.append(nn.ReLU())
		else:
			layers.append(nn.Linear(input_dim, width))
			layers.append(nn.ReLU())
		for i in range(depth):
			layers.append(nn.Linear(width, width))
			layers.append(nn.ReLU())

		self.encoder = nn.ModuleList(encoding_layers)
		self.base = nn.ModuleList(layers)

		# Initialize template mesh to be example foot
		if template_mesh_loc is not None:
			verts, face_dict, props = p3d_load_obj(template_mesh_loc)
			faces = face_dict.verts_idx

			# Compute average colour
			mesh = Meshes(verts.unsqueeze(0).to(device), faces.unsqueeze(0).to(device),
						  textures=TexturesUV(props.texture_images['material_0'].unsqueeze(0).to(device),
											  faces_uvs=face_dict.textures_idx.unsqueeze(0).to(device),
											  verts_uvs=props.verts_uvs.unsqueeze(0).to(device)))

			# Shift centroid to origin
			shift = torch.mean(verts, dim=0)
			verts = verts - shift

			_, samples = sample_points_from_meshes(mesh, num_samples=1000, return_textures=True)
			avg_col = samples.mean(dim=(0, 1))

		else:
			verts = torch.zeros((1, 3), dtype=torch.float32)
			faces = torch.ones((1, 3), dtype=torch.int)
			avg_col = torch.zeros(3, dtype=torch.float32)

		self.template_verts = nn.Parameter(verts.unsqueeze(0).float().to(device), requires_grad=False)
		self.template_faces = nn.Parameter(faces.unsqueeze(0).to(device), requires_grad=False)
		self.avg_col = nn.Parameter(avg_col.to(device), requires_grad=False)

		self.template_mesh = Meshes(verts=self.template_verts, faces=self.template_faces).to(device)

		# Load texture to identify which faces need to be masked out during the rendering cycle
		# self.masked_faces = torch.tensor([]).to(device)
		# # if (u, v) = (0, 0) for final UV vertex, that means that some of the faces are set to be masked out. Find and store these
		# if (self.template_tex.verts_uvs_padded()[0, -1] == 0).all():
		# 	face_uv = self.template_tex.faces_uvs_padded()[0]
		# 	vt = self.template_tex.verts_uvs_padded().shape[1] - 1 # index corresponding to this final vertex
		# 	self.masked_faces = torch.argwhere(torch.all(face_uv==vt, dim=-1)).flatten()

		self.latent_vectors_train, self.latent_vectors_val = [], []

		self.shapevec_size = shapevec_size
		self.shapevec, self.shapevec_val = None, None
		self.use_shapevec = use_shapevec
		if use_shapevec:
			labels = None if latent_labels is None else latent_labels.get('shape', None)
			labels_val = None if latent_labels is None else latent_labels.get('shape_val', None)
			self.shapevec = LatentVector(train_size, vec_size=shapevec_size, name='shapevec_train', key='shape',
										 labels=labels, device=device)
			self.shapevec_val = LatentVector(val_size, vec_size=shapevec_size, name='shapevec_val', key='shape',
											 labels=labels_val, device=device)
			self.latent_vectors_train += [self.shapevec]
			self.latent_vectors_val += [self.shapevec_val]

		self.posevec_size = posevec_size
		self.posevec, self.posevec_val = None, None
		self.use_posevec = use_posevec
		if use_posevec:
			labels = None if latent_labels is None else latent_labels.get('pose', None)
			labels_val = None if latent_labels is None else latent_labels.get('pose_val', None)
			self.posevec = LatentVector(train_size, vec_size=shapevec_size, name='posevec_train', key='pose',
										labels=labels, device=device)
			self.posevec_val = LatentVector(val_size, vec_size=shapevec_size, name='posevec_val', key='pose',
											labels=labels_val, device=device)
			self.latent_vectors_train += [self.posevec]
			self.latent_vectors_val += [self.posevec_val]

		self.texvec_size = texvec_size
		self.texvec, self.texvec_val = None, None
		self.use_texvec = use_texvec
		if use_texvec:
			labels = None if latent_labels is None else latent_labels.get('tex', None)
			labels_val = None if latent_labels is None else latent_labels.get('tex_val', None)
			self.texvec = LatentVector(train_size, vec_size=texvec_size, name='texvec_train', key='tex', labels=labels,
									   device=device)
			self.texvec_val = LatentVector(val_size, vec_size=texvec_size, name='texvec_val', key='tex',
										   labels=labels_val, device=device)
			self.latent_vectors_train += [self.texvec]
			self.latent_vectors_val += [self.texvec_val]

		# Learn a per image registration (shift & euler angles)
		labels = None if latent_labels is None else latent_labels.get('reg', None)
		labels_val = None if latent_labels is None else latent_labels.get('reg_val', None)
		self.reg = LatentVector(train_size, vec_size=9, name='reg_train', device=device, key='reg', labels=labels,
								init_values=np.array([0] * 6 + [1] * 3))
		self.reg_val = LatentVector(val_size, vec_size=9, name='reg_val', device=device, key='reg', labels=labels_val,
									init_values=np.array([0] * 6 + [1] * 3))
		self.latent_vectors_train += [self.reg]
		self.latent_vectors_val += [self.reg_val]

		# Branches
		disp_layers = []
		disp_input_size = width + self.shapevec_size * self.use_texvec + self.posevec_size * self.use_posevec
		for i in range(dispdepth):
			width_0 = width if i > 0 else disp_input_size
			disp_layers.append(nn.Linear(width_0, width))
			disp_layers.append(nn.ReLU())
		disp_layers.append(nn.Linear(width, 3))
		self.mlp_disp = nn.Sequential(*disp_layers)

		col_layers = []
		col_input_size = width + self.texvec_size * self.use_texvec
		for i in range(coldepth):
			width_0 = width if i > 0 else col_input_size
			col_layers.append(nn.Linear(width_0, width))
			col_layers.append(nn.ReLU())

		self.use_avg_colour = use_avg_colour

		ncols = 3
		col_layers.append(nn.Linear(width, ncols))
		self.mlp_col = nn.Sequential(*col_layers)

		main_params = [self.base, self.mlp_disp, self.mlp_col, self.shapevec, self.texvec, self.posevec]

		self.restyle_features_per_vertex = restyle_features_per_vertex

		# Load template per-vertex features if given
		self.per_vertex_features = None
		if opts is not None and opts.template_features_pth is not None:
			self.per_vertex_features = nn.Parameter(
				load_template_classes(opts.template_features_pth, device=device).unsqueeze(0))

		self.main_params = make_params_list(*main_params)
		self.templ_params = make_params_list(self.template_verts, self.reg)
		self.val_params = make_params_list(self.shapevec_val, self.texvec_val, self.posevec_val)
		self.reg_params = make_params_list(self.reg, self.reg_val)
		self.latent_params = make_params_list(self.shapevec, self.shapevec_val, self.texvec, self.texvec_val,
											  self.posevec, self.posevec_val)

		self.reset_weights()
		self.onnx_mode = False  # Turn this on for saving to web format

	def forward(self, pos, shapevec=None, texvec=None, posevec=None):
		"""
		:param pos: [Batch x Num points x 3] (x,y,z) coordinates
		:param shapevec: [Batch x vector_size] Latent texture representation
		:param onnx_model: If True, compute everything in [Batch x 3] shape instead (as other is not compatible with web operations)
		:return: dict of disp [Batch x Num points x 3], col [Batch x Num points x 3]
		"""

		# Perform reshaping
		if not self.onnx_mode:
			batch, np, _ = pos.shape
			if batch == 1 and shapevec is not None:  # In case of no batch for pos, but batch for vectors
				batch = shapevec.shape[0]
				pos = pos.expand(batch, -1, -1)

			if shapevec is not None:
				shapevec = shapevec.unsqueeze(1).expand(-1, np, -1)  # Add points dimension to shape

			if posevec is not None:
				posevec = posevec.unsqueeze(1).expand(-1, np, -1)  # Add points dimension to pose

			if texvec is not None:
				texvec = texvec.unsqueeze(1).expand(-1, np, -1)  # Add points dimension to texture

		else:
			batch = pos.shape[0];
			np = 1

		for layer in self.encoder:
			pos = layer(pos)  # Encode position

		x = pos
		for layer in self.base:
			x = layer(x)

		disp_input = x
		if shapevec is not None:
			disp_input = torch.cat([disp_input, shapevec], dim=-1)

		if posevec is not None:
			disp_input = torch.cat([disp_input, posevec], dim=-1)

		col_input = x
		if texvec is not None:
			col_input = torch.cat([col_input, texvec], dim=-1)

		disp = self.mlp_disp(disp_input)
		col = self.mlp_col(col_input)

		out = {}

		out['disp'] = 0.1 * torch.tanh(disp)

		if self.use_avg_colour:
			col = self.avg_col[None, None, :] + 0.5 * (1 + torch.tanh(col))
		else:
			col = 0.5 * (1 + torch.tanh(col))

		out['col'] = col

		return out

	def get_meshes(self, shapevec=None, reg=None, texvec=None, posevec=None, no_displacement=False,
				   include_texture=True):
		"""
		From template mesh, sample *all* points.
		Apply shapevec, calculate offsets
		Apply global registration.
		Return transformed Meshes

		:param shapevec: [ N x shapevec_size ]
		:param reg: [ N x 6 ] Per-instance registration & scaling
		no_displacement: Flag to not apply offsets to template (for experimental reasons)
		:return: dict(mesh = Meshes, offsets=offsets from template)
		"""
		if shapevec is None:
			N = 0
		else:
			N, _ = shapevec.shape

		meshes = extend_template(self.template_mesh, N=N)

		# Sample offsets
		verts = meshes.verts_padded()
		res = self(verts, shapevec=shapevec, texvec=texvec, posevec=posevec)
		offsets, col = res['disp'], res['col']

		# Apply learned reg & scale
		if reg is not None:
			S = reg[..., 6:9]
			centroid_offset = reg[..., :3]
			euler_rot = reg[..., 3:6]
			R = euler_angles_to_matrix(euler_rot, 'XYZ')

			T = Transform3d(device=S.device).scale(S).rotate(R).translate(centroid_offset)
			X = T.transform_points(verts + offsets)

		else:
			X = verts + offsets

		if not no_displacement:
			meshes = meshes.update_padded(X)

		if self.use_texvec:
			meshes.textures = TexturesVertex(col[..., :3])
		else:
			meshes.textures = self.template_tex.extend(N)

		if self.per_vertex_features is not None:
			res['cpv'] = self.per_vertex_features

		return dict(meshes=meshes, offsets=offsets, verts=X, **res)

	def get_meshes_from_batch(self, batch, is_train=True, no_displacement=False):
		svec, tvec, pvec, reg = [f"{a}_{['val', 'train'][is_train]}" for a in ['shapevec', 'texvec', 'posevec', 'reg']]
		res = self.get_meshes(shapevec=batch.get(svec, None),
							  reg=batch.get(reg, None),
							  texvec=batch.get(tvec, None),
							  posevec=batch.get(pvec, None),
							  include_texture=True,
							  no_displacement=no_displacement)
		return res

	def reset_weights(self):
		self.mlp_disp[-1].weight.data.zero_()
		self.mlp_disp[-1].bias.data.zero_()

	def configure_template(self, state_dict, device='cuda'):
		verts = state_dict['template_verts']
		faces = state_dict['template_faces']

		self.template_verts = nn.Parameter(verts.float().to(device), requires_grad=False)
		self.template_faces = nn.Parameter(faces.to(device), requires_grad=False)
		self.template_mesh = Meshes(verts=self.template_verts, faces=self.template_faces).to(device)

		if 'avg_col' in state_dict:
			avg_col = state_dict['avg_col']
			self.avg_col = nn.Parameter(avg_col.to(device), requires_grad=False)

	def to(self, device):
		self.template_mesh = self.template_mesh.to(device)
		return super().to(device)

class PCAModel(Model):
	"""Linear PCA Model"""

	def __init__(self, *args, train_size=None, val_size=None, device='cuda', **kwargs):
		super().__init__()

		self.params = dict(train_size=train_size, val_size=val_size)
		self.train_size = train_size
		self.val_size = val_size

		# Configured in configure_template function
		self.template_verts = None
		self.template_faces = None
		self.template_mesh = None

		# V verts, B shape coeffs
		self.pca_var = None  # [B]
		self.pca_coefs = None  # [V x B x 3]

		self.latent_vectors_train, self.latent_vectors_val = [], []
		self.shapevec, self.shapevec_val = None, None

		self.reg = LatentVector(train_size, vec_size=9, name='reg_train', device=device,
								init_values=np.array([0] * 6 + [1] * 3))
		self.reg_val = LatentVector(val_size, vec_size=9, name='reg_val', device=device,
									init_values=np.array([0] * 6 + [1] * 3))
		self.latent_vectors_train += [self.reg]
		self.latent_vectors_val += [self.reg_val]

		self.configure_params()

	def configure_params(self):
		self.main_params = make_params_list(self.shapevec)
		self.val_params = make_params_list(self.shapevec_val)
		self.reg_params = make_params_list(self.reg, self.reg_val)
		self.latent_params = make_params_list(self.shapevec, self.shapevec_val)

	def get_meshes(self, shapevec=None, reg=None, **kwargs):

		N, _ = shapevec.shape
		meshes = extend_template(self.template_mesh, N=N)

		# Sample offsets
		verts = meshes.verts_padded()

		offsets = (self.pca_coefs.unsqueeze(0) * shapevec.unsqueeze(1).unsqueeze(-1)).sum(dim=2)

		# Apply learned reg & scale
		if reg is not None:
			S = reg[..., 6:9]
			centroid_offset = reg[..., :3]
			euler_rot = reg[..., 3:6]
			R = euler_angles_to_matrix(euler_rot, 'XYZ')

			T = Transform3d(device=S.device).scale(S).rotate(R).translate(centroid_offset)
			X = T.transform_points(verts + offsets)

		else:
			X = verts + offsets

		meshes = meshes.update_padded(X)

		col = torch.full(X.shape, 0.5).float().to(meshes.device)  # set all texture colour to grey
		meshes.textures = TexturesVertex(col)

		return dict(meshes=meshes, offsets=offsets, verts=X)

	def get_meshes_from_batch(self, batch, is_train=True, **kwargs):
		svec, reg = [f"{a}_{['val', 'train'][is_train]}" for a in ['shapevec', 'reg']]
		res = self.get_meshes(shapevec=batch.get(svec, None),
							  reg=batch.get(reg, None))

		return res

	@classmethod
	def load_from_mat(cls, src, device, **kwargs):
		"""Load data from .mat file"""
		data = loadmat(src)

		state_dict = {}
		state_dict['template_verts'] = torch.from_numpy(data['pcaMean'].reshape(-1, 3)).unsqueeze(0).float()
		state_dict['template_faces'] = torch.from_numpy(data['mesh']).unsqueeze(0).long() - 1

		V = state_dict['template_verts'].shape[1]
		state_dict['pca_coefs'] = torch.from_numpy(data['pcaCoefs']).reshape(V, 3, -1).permute(0, 2,
																							   1).float()  # V x B x 3
		state_dict['pca_var'] = torch.from_numpy(data['pcaVar']).float()

		model = cls(device=device, **kwargs)
		model.configure_template(state_dict, device=device)
		model.load_state_dict(state_dict, strict=False)
		model.configure_params()
		return model

	@classmethod
	def load(cls, file, device='cuda', opts=None, **kwargs):
		if file.endswith('.pth'):
			return super().load(file, device=device, opts=opts, **kwargs)
		elif file.endswith('.mat'):
			return cls.load_from_mat(file, device=device, opts=opts, **kwargs)
		else:
			raise NotImplementedError(f"Filetype `{os.path.splitext(file)[-1]}` for PCA model not understood")

	def configure_template(self, state_dict, device='cuda'):
		verts = state_dict['template_verts']
		faces = state_dict['template_faces']

		self.template_verts = nn.Parameter(verts.float().to(device), requires_grad=False)
		self.template_faces = nn.Parameter(faces.to(device), requires_grad=False)
		self.template_mesh = Meshes(verts=self.template_verts, faces=self.template_faces).to(device)

		self.pca_var = nn.Parameter(state_dict['pca_var'].to(device), requires_grad=False)  # [B]
		self.pca_coefs = nn.Parameter(state_dict['pca_coefs'].to(device), requires_grad=False)  # [V x B x 3]

		V, B, _ = self.pca_coefs.shape

		self.shapevec = LatentVector(self.train_size, vec_size=B, name='shapevec_train', device=device)
		self.shapevec_val = LatentVector(self.val_size, vec_size=B, name='shapevec_val', device=device)
		self.latent_vectors_train += [self.shapevec]
		self.latent_vectors_val += [self.shapevec_val]


class VertexFeaturesModel(Model):
	def __init__(self, template_mesh_loc=None, device='cuda',
				 restyle_features_per_vertex=False, restyle_cluster_per_vertex=False,
				 feature_size=21, feature_key='cpv', **kwargs):
		"""Model predicts features per vertex on template mesh
		"""

		super(VertexFeaturesModel, self).__init__()

		self.params = dict(restyle_features_per_vertex=restyle_features_per_vertex,
						   restyle_cluster_per_vertex=restyle_cluster_per_vertex,
						   feature_size=feature_size, feature_key=feature_key)

		self.feature_key = feature_key
		masked_faces = torch.zeros(1, 3).to(device)

		# Initialize template mesh to be example foot
		if template_mesh_loc is not None:
			verts, face_dict, props = p3d_load_obj(template_mesh_loc)
			faces = face_dict.verts_idx

			# Shift centroid to origin
			shift = torch.mean(verts, dim=0)
			verts = verts - shift

			# Compute average colour
			template_tex = TexturesUV(props.texture_images['material_0'].unsqueeze(0).to(device),
									  faces_uvs=face_dict.textures_idx.unsqueeze(0).to(device),
									  verts_uvs=props.verts_uvs.unsqueeze(0).to(device))
			mesh = Meshes(verts.unsqueeze(0).to(device), faces.unsqueeze(0).to(device), textures=template_tex)

			_, samples = sample_points_from_meshes(mesh, num_samples=1000, return_textures=True)
			avg_col = samples.mean(dim=(0, 1))

			self.masked_faces = torch.tensor([]).to(device)
			# if (u, v) = (0, 0) for final UV vertex, that means that some of the faces are set to be masked out. Find and store these
			if (template_tex.verts_uvs_padded()[0, -1] == 0).all():
				face_uv = template_tex.faces_uvs_padded()[0]
				vt = template_tex.verts_uvs_padded().shape[1] - 1  # index corresponding to this final vertex
				masked_faces = torch.argwhere(torch.all(face_uv == vt, dim=-1)).flatten()

		else:
			verts = torch.zeros((1, 3), dtype=torch.float32)
			faces = torch.ones((1, 3), dtype=torch.int)
			avg_col = torch.zeros(3, dtype=torch.float32)

		self.template_verts = nn.Parameter(verts.unsqueeze(0).float().to(device), requires_grad=False)
		self.template_faces = nn.Parameter(faces.unsqueeze(0).to(device), requires_grad=False)
		self.avg_col = nn.Parameter(avg_col.to(device), requires_grad=False)
		self.masked_faces = nn.Parameter(masked_faces, requires_grad=False)

		V, _ = verts.shape
		self.features = nn.Parameter(torch.zeros(V, feature_size).to(device), requires_grad=True)
		self.colours = nn.Parameter(torch.full((V, 3), 0.5).unsqueeze(0).to(device), requires_grad=True)
		self.deforms = nn.Parameter(torch.zeros(V, 3).to(device), requires_grad=True)
		self.template_mesh = Meshes(verts=self.template_verts, faces=self.template_faces).to(device)

		self.reg = LatentVector(1, vec_size=9, name='reg', device=device, init_values=np.array([0] * 6 + [1] * 3))
		# self.vertex_cols = nn.Parameter(torch.zeros(V, 3).to(device), requires_grad=True)

		main_params = [self.features, self.colours]  # , self.vertex_cols] #, self.deforms]

		self._blank_param = nn.Parameter(torch.zeros(1), requires_grad=True)

		self.latent_vectors_train = []
		self.latent_vectors_val = []

		self.main_params = make_params_list(*main_params)  # For training
		self.templ_params = make_params_list(self._blank_param)
		self.val_params = make_params_list(self._blank_param)
		self.reg_params = make_params_list(self.reg)
		self.latent_params = make_params_list(self._blank_param)

	@property
	def inference_params(self):
		return make_params_list(self.deforms, self.reg)

	def get_meshes(self, reg=None, no_displacement=False, *args, **kwargs):
		if reg is None:
			N = 1
		else:
			N, _ = reg.shape

		meshes = extend_template(self.template_mesh, N=N)
		verts = meshes.verts_padded()

		out = {}
		out[self.feature_key] = self.features.unsqueeze(0)

		# Apply learned reg & scale
		if reg is not None:
			S = reg[..., 6:9]
			centroid_offset = reg[..., :3]
			euler_rot = reg[..., 3:6]
			R = euler_angles_to_matrix(euler_rot, 'XYZ')

			T = Transform3d(device=S.device).scale(S).rotate(R).translate(centroid_offset)
			X = T.transform_points(verts + self.deforms)

		else:
			X = verts + self.deforms

		if not no_displacement:
			meshes = meshes.update_padded(X)

		# col = torch.full(verts.shape, 0.5).float().to(meshes.device) # set all texture colour to grey
		col = self.colours
		meshes.textures = TexturesVertex(col)

		out['meshes'] = meshes

		return out

	def get_meshes_from_batch(self, batch, is_train=True, *args, **kwargs):
		reg_key = f"reg_{['val', 'train'][is_train]}"
		res = self.get_meshes(reg=batch.get(reg_key, None), *args, **kwargs)
		return res

	def configure_template(self, state_dict, device='cuda'):
		verts = state_dict['template_verts']
		faces = state_dict['template_faces']
		feats = state_dict['features']
		deforms = state_dict['deforms']
		colours = state_dict['colours']
		masked_faces = state_dict['masked_faces']

		self.template_verts = nn.Parameter(verts.float().to(device), requires_grad=False)
		self.template_faces = nn.Parameter(faces.to(device), requires_grad=False)
		self.template_mesh = Meshes(verts=self.template_verts, faces=self.template_faces).to(device)
		self.features = nn.Parameter(feats.to(device), requires_grad=True)
		self.deforms = nn.Parameter(deforms.to(device), requires_grad=True)
		self.colours = nn.Parameter(colours.to(device), requires_grad=True)
		self.masked_faces = nn.Parameter(masked_faces.to(device), requires_grad=False)

		if 'avg_col' in state_dict:
			avg_col = state_dict['avg_col']
			self.avg_col = nn.Parameter(avg_col.to(device), requires_grad=False)


model_zoo = dict(neural=NeuralDisplacementField, pca=PCAModel, vertexfeatures=VertexFeaturesModel)


def process_opts(opts: Union[Opts, str], eval=False):
	if isinstance(opts, str):
		opts = Opts.load(opts)

	if eval:
		opts.dont_load_latents = False  # Always override this setting if in eval mode

	return opts


def model_class_from_opts(opts: Union[Opts, str]):
	"""Return the class instance corresponding to the model_type in opts file.
	
	opts: either Opts object, or .yaml file location from which Opts can be loaded"""
	opts = process_opts(opts)
	return model_zoo[opts.model_type]


def model_from_opts(opts: Union[Opts, str]):
	"""Return a loaded model from a given opts file"""
	opts = process_opts(opts)
	model_class = model_class_from_opts(opts)
	return model_class.load(opts.load_model, device=opts.device, opts=opts)
