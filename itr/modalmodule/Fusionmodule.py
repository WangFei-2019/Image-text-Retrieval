import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .utils import l1norm, l2norm


# ---------------------- VSRN ----------------------------
class S2VTAttModel(nn.Module):
	def __init__(self, encoder, decoder):
		"""
		Args:
			encoder (nn.Module): Encoder rnn
			decoder (nn.Module): Decoder rnn
		"""
		super(S2VTAttModel, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, vid_feats, target_variable=None,
				mode='train', config: dict = {}):
		"""
		Args:
			vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
			target_variable (None, configional): groung truth labels

		Returns:
			seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
			seq_preds: [] or Variable of shape [batch_size, max_len-1]
		"""
		encoder_outputs, encoder_hidden = self.encoder(vid_feats)
		seq_prob, seq_preds = self.decoder(encoder_outputs, encoder_hidden, target_variable, mode, config)
		return seq_prob, seq_preds


class S2VTModel(nn.Module):
	def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=2048, sos_id=1, eos_id=0,
				 n_layers=1, rnn_cell='gru', rnn_dropout_p=0.2):
		super(S2VTModel, self).__init__()
		if rnn_cell.lower() == 'lstm':
			self.rnn_cell = nn.LSTM
		elif rnn_cell.lower() == 'gru':
			self.rnn_cell = nn.GRU
		self.rnn1 = self.rnn_cell(dim_vid, dim_hidden, n_layers,
								  batch_first=True, dropout=rnn_dropout_p)
		self.rnn2 = self.rnn_cell(dim_hidden + dim_word, dim_hidden, n_layers,
								  batch_first=True, dropout=rnn_dropout_p)

		self.dim_vid = dim_vid
		self.dim_output = vocab_size
		self.dim_hidden = dim_hidden
		self.dim_word = dim_word
		self.max_length = max_len
		self.sos_id = sos_id
		self.eos_id = eos_id
		self.embedding = nn.Embedding(self.dim_output, self.dim_word)

		self.out = nn.Linear(self.dim_hidden, self.dim_output)

	def forward(self, vid_feats, target_variable=None,
				mode='train', config={}):
		batch_size, n_frames, _ = vid_feats.shape
		padding_words = vid_feats.data.new(batch_size, n_frames, self.dim_word).zero_()
		padding_frames = vid_feats.data.new(batch_size, 1, self.dim_vid).zero_()
		state1 = None
		state2 = None
		# self.rnn1.flatten_parameters()
		# self.rnn2.flatten_parameters()
		output1, state1 = self.rnn1(vid_feats, state1)
		input2 = torch.cat((output1, padding_words), dim=2)
		output2, state2 = self.rnn2(input2, state2)

		seq_probs = []
		seq_preds = []
		if mode == 'train':
			for i in range(self.max_length - 1):
				# <eos> doesn't input to the network
				current_words = self.embedding(target_variable[:, i])
				self.rnn1.flatten_parameters()
				self.rnn2.flatten_parameters()
				output1, state1 = self.rnn1(padding_frames, state1)
				input2 = torch.cat(
					(output1, current_words.unsqueeze(1)), dim=2)
				output2, state2 = self.rnn2(input2, state2)
				logits = self.out(output2.squeeze(1))
				logits = F.log_softmax(logits, dim=1)
				seq_probs.append(logits.unsqueeze(1))
			seq_probs = torch.cat(seq_probs, 1)

		else:
			current_words = self.embedding(
				Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda())
			for i in range(self.max_length - 1):
				self.rnn1.flatten_parameters()
				self.rnn2.flatten_parameters()
				output1, state1 = self.rnn1(padding_frames, state1)
				input2 = torch.cat(
					(output1, current_words.unsqueeze(1)), dim=2)
				output2, state2 = self.rnn2(input2, state2)
				logits = self.out(output2.squeeze(1))
				logits = F.log_softmax(logits, dim=1)
				seq_probs.append(logits.unsqueeze(1))
				_, preds = torch.max(logits, 1)
				current_words = self.embedding(preds)
				seq_preds.append(preds.unsqueeze(1))
			seq_probs = torch.cat(seq_probs, 1)
			seq_preds = torch.cat(seq_preds, 1)
		return seq_probs, seq_preds


class Attention(nn.Module):
	"""
	Applies an attention mechanism on the output features from the decoder.
	"""

	def __init__(self, dim):
		super(Attention, self).__init__()
		self.dim = dim
		self.linear1 = nn.Linear(dim * 2, dim)
		self.linear2 = nn.Linear(dim, 1, bias=False)

	# self._init_hidden()

	def _init_hidden(self):
		nn.init.xavier_normal_(self.linear1.weight)
		nn.init.xavier_normal_(self.linear2.weight)

	def forward(self, hidden_state, encoder_outputs):
		"""
		Arguments:
			hidden_state {Variable} -- batch_size x dim
			encoder_outputs {Variable} -- batch_size x seq_len x dim

		Returns:
			Variable -- context vector of size batch_size x dim
		"""
		batch_size, seq_len, _ = encoder_outputs.size()
		hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
		inputs = torch.cat((encoder_outputs, hidden_state),
						   2).view(-1, self.dim * 2)
		o = self.linear2(F.tanh(self.linear1(inputs)))
		e = o.view(batch_size, seq_len)
		alpha = F.softmax(e, dim=1)
		context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
		return context


class EncoderRNN(nn.Module):
	def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
				 n_layers=1, bidirectional=False, rnn_cell='gru'):
		"""
		Args:
			hidden_dim (int): dim of hidden state of rnn
			input_dropout_p (int): dropout probability for the input sequence
			dropout_p (float): dropout probability for the output sequence
			n_layers (int): number of rnn layers
			rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
		"""
		super(EncoderRNN, self).__init__()
		self.dim_vid = dim_vid
		self.dim_hidden = dim_hidden
		self.input_dropout_p = input_dropout_p
		self.rnn_dropout_p = rnn_dropout_p
		self.n_layers = n_layers
		self.bidirectional = bidirectional
		self.rnn_cell = rnn_cell

		self.vid2hid = nn.Linear(dim_vid, dim_hidden)
		self.input_dropout = nn.Dropout(input_dropout_p)

		if rnn_cell.lower() == 'lstm':
			self.rnn_cell = nn.LSTM
		elif rnn_cell.lower() == 'gru':
			self.rnn_cell = nn.GRU

		self.rnn = self.rnn_cell(input_size=dim_hidden, hidden_size=dim_hidden, num_layers=n_layers,
								 batch_first=True,
								 bidirectional=bool(bidirectional), dropout=self.rnn_dropout_p)

		self._init_hidden()

	def _init_hidden(self):
		nn.init.xavier_normal_(self.vid2hid.weight)

	def forward(self, vid_feats):
		"""
		Applies a multi-layer RNN to an input sequence.
		Args:
			input_var (batch, seq_len): tensor containing the features of the input sequence.
			input_lengths (list of int, optional): A list that contains the lengths of sequences
			  in the mini-batch
		Returns: output, hidden
			- **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
			- **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
		"""
		batch_size, seq_len, dim_vid = vid_feats.size()
		vid_feats = self.vid2hid(vid_feats.reshape(-1, dim_vid))
		vid_feats = self.input_dropout(vid_feats)
		vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
		self.rnn.flatten_parameters()
		output, hidden = self.rnn(input=vid_feats)
		return output, hidden


class DecoderRNN(nn.Module):
	"""
	Provides functionality for decoding in a seq2seq framework, with an configion for attention.
	Args:
		vocab_size (int): size of the vocabulary
		max_len (int): a maximum allowed length for the sequence to be processed
		dim_hidden (int): the number of features in the hidden state `h`
		n_layers (int, configional): number of recurrent layers (default: 1)
		rnn_cell (str, configional): type of RNN cell (default: gru)
		bidirectional (bool, configional): if the encoder is bidirectional (default False)
		input_dropout_p (float, configional): dropout probability for the input sequence (default: 0)
		rnn_dropout_p (float, configional): dropout probability for the output sequence (default: 0)

	"""

	def __init__(self,
				 vocab_size,
				 max_len,
				 dim_hidden,
				 dim_word,
				 n_layers=1,
				 rnn_cell='gru',
				 bidirectional=False,
				 input_dropout_p=0.1,
				 rnn_dropout_p=0.1):
		super(DecoderRNN, self).__init__()

		self.bidirectional_encoder = bidirectional

		self.dim_output = vocab_size
		self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
		self.dim_word = dim_word
		self.max_length = max_len
		self.sos_id = 1
		self.eos_id = 0
		self.input_dropout = nn.Dropout(input_dropout_p)
		self.embedding = nn.Embedding(self.dim_output, dim_word)
		self.attention = Attention(self.dim_hidden)
		if rnn_cell.lower() == 'lstm':
			self.rnn_cell = nn.LSTM
		elif rnn_cell.lower() == 'gru':
			self.rnn_cell = nn.GRU
		self.rnn = self.rnn_cell(
			input_size=self.dim_hidden + dim_word,
			hidden_size=self.dim_hidden,
			num_layers=n_layers,
			batch_first=True,
			dropout=rnn_dropout_p)

		self.out = nn.Linear(self.dim_hidden, self.dim_output)

		self._init_weights()

	def forward(self,
				encoder_outputs,
				encoder_hidden,
				targets=None,
				mode='train',
				config: dict = {}):
		"""

		Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
		- **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
		  hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
		- **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
		- **targets** (batch, max_length): targets labels of the ground truth sentences

		Outputs: seq_probs,
		- **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
		- **seq_preds** (batch_size, max_length): predicted symbols
		"""
		sample_max = config.get('sample_max', 1)
		beam_size = config.get('beam_size', 1)
		temperature = config.get('temperature', 1.0)

		batch_size, _, _ = encoder_outputs.size()
		decoder_hidden = self._init_rnn_state(encoder_hidden)

		seq_logprobs = []
		seq_preds = []
		self.rnn.flatten_parameters()
		if mode == 'train':
			# use targets as rnn inputs
			targets_emb = self.embedding(targets)
			for i in range(self.max_length - 1):
				current_words = targets_emb[:, i, :]
				context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
				decoder_input = torch.cat([current_words, context], dim=1)
				decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
				decoder_output, decoder_hidden = self.rnn(
					decoder_input, decoder_hidden)
				logprobs = F.log_softmax(
					self.out(decoder_output.squeeze(1)), dim=1)
				seq_logprobs.append(logprobs.unsqueeze(1))

			seq_logprobs = torch.cat(seq_logprobs, 1)

		elif mode == 'inference':
			if beam_size > 1:
				return self.sample_beam(encoder_outputs, decoder_hidden, config)

			for t in range(self.max_length - 1):
				context = self.attention(
					decoder_hidden.squeeze(0), encoder_outputs)

				if t == 0:  # input <bos>
					it = torch.LongTensor([self.sos_id] * batch_size).cuda()
				elif sample_max:
					sampleLogprobs, it = torch.max(logprobs, 1)
					seq_logprobs.append(sampleLogprobs.view(-1, 1))
					it = it.view(-1).long()

				else:
					# sample according to distribuition
					if temperature == 1.0:
						prob_prev = torch.exp(logprobs)
					else:
						# scale logprobs by temperature
						prob_prev = torch.exp(torch.div(logprobs, temperature))
					it = torch.multinomial(prob_prev, 1).cuda()
					sampleLogprobs = logprobs.gather(1, it)
					seq_logprobs.append(sampleLogprobs.view(-1, 1))
					it = it.view(-1).long()

				seq_preds.append(it.view(-1, 1))

				xt = self.embedding(it)
				decoder_input = torch.cat([xt, context], dim=1)
				decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
				decoder_output, decoder_hidden = self.rnn(
					decoder_input, decoder_hidden)
				logprobs = F.log_softmax(
					self.out(decoder_output.squeeze(1)), dim=1)

			seq_logprobs = torch.cat(seq_logprobs, 1)
			seq_preds = torch.cat(seq_preds[1:], 1)

		return seq_logprobs, seq_preds

	def _init_weights(self):
		""" init the weight of some layers
		"""
		nn.init.xavier_normal_(self.out.weight)

	def _init_rnn_state(self, encoder_hidden):
		""" Initialize the encoder hidden state. """
		if encoder_hidden is None:
			return None
		if isinstance(encoder_hidden, tuple):
			encoder_hidden = tuple(
				[self._cat_directions(h) for h in encoder_hidden])
		else:
			encoder_hidden = self._cat_directions(encoder_hidden)
		return encoder_hidden

	def _cat_directions(self, h):
		""" If the encoder is bidirectional, do the following transformation.
			(#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
		"""
		if self.bidirectional_encoder:
			h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
		return h


# ---------------------------------------------------------

# ---------------------- SGRAF ----------------------------
class EncoderSimilarity(nn.Module):
	"""
	Compute the image-text similarity by SGR, SAF, AVE
	Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
		  - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
	Returns:
		- sim_all: final image-text similarities, shape: (batch_size, batch_size).
	"""

	def __init__(self, embed_size, sim_dim, module_name='AVE', sgr_step=3):
		super(EncoderSimilarity, self).__init__()
		self.module_name = module_name

		self.v_global_w = VisualSA(embed_size, 0.4, 36)
		self.t_global_w = TextSA(embed_size, 0.4)

		self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
		self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)

		self.sim_eval_w = nn.Linear(sim_dim, 1)
		self.sigmoid = nn.Sigmoid()

		if module_name == 'SGR':
			self.SGR_module = nn.Sequential()
			for i in range(sgr_step):
				self.SGR_module.add_module(f'sgr{i}', GraphReasoning(sim_dim))
		elif module_name == 'SAF':
			self.SAF_module = AttentionFiltration(sim_dim)
		else:
			raise ValueError('Invalid input of config.module_name in configs.py')

		self.init_weights()

	def forward(self, img_emb, cap_emb, cap_lens, *args, **kwargs):
		sim_all = []
		n_image = img_emb.size(0)
		n_caption = cap_emb.size(0)

		# get enhanced global images by self-attention
		img_ave = torch.mean(img_emb, 1)
		img_glo = self.v_global_w(img_emb, img_ave)

		for i in range(n_caption):
			# get the i-th sentence
			n_word = cap_lens[i]
			cap_i = cap_emb[i, :n_word, :].unsqueeze(0)

			cap_i_expand = cap_i.repeat(n_image, 1, 1)

			# get enhanced global i-th text by self-attention
			cap_ave_i = torch.mean(cap_i, 1)
			cap_glo_i = self.t_global_w(cap_i, cap_ave_i)

			# local-global alignment construction
			Context_img = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)
			sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)
			sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)

			sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)
			sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)

			# concat the global and local alignments
			sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)

			# compute the final similarity vector
			if self.module_name == 'SGR':
				sim_emb = self.SGR_module(sim_emb)
				sim_vec = sim_emb[:, 0, :]
			else:
				sim_vec = self.SAF_module(sim_emb)

			# compute the final similarity score
			sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
			sim_all.append(sim_i)

		# (n_image, n_caption)
		sim_all = torch.cat(sim_all, 1)

		return sim_all

	def init_weights(self):
		for m in self.children():
			if isinstance(m, nn.Linear):
				r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
				m.weight.data.uniform_(-r, r)
				m.bias.data.fill_(0)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


class VisualSA(nn.Module):
	"""
	Build global image representations by self-attention.
	Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
		  - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
	Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
	"""

	def __init__(self, embed_dim, dropout_rate, num_region):
		super(VisualSA, self).__init__()

		self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
											 nn.BatchNorm1d(num_region),
											 nn.Tanh(), nn.Dropout(dropout_rate))
		self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
											  nn.BatchNorm1d(embed_dim),
											  nn.Tanh(), nn.Dropout(dropout_rate))
		self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

		self.init_weights()
		self.softmax = nn.Softmax(dim=1)

	def init_weights(self):
		for embeddings in self.children():
			for m in embeddings:
				if isinstance(m, nn.Linear):
					r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
					m.weight.data.uniform_(-r, r)
					m.bias.data.fill_(0)
				elif isinstance(m, nn.BatchNorm1d):
					m.weight.data.fill_(1)
					m.bias.data.zero_()

	def forward(self, local, raw_global):
		# compute embedding of local regions and raw global image
		l_emb = self.embedding_local(local)
		g_emb = self.embedding_global(raw_global)

		# compute the normalized weights, shape: (batch_size, 36)
		g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
		common = l_emb.mul(g_emb)
		weights = self.embedding_common(common).squeeze(2)
		weights = self.softmax(weights)

		# compute final image, shape: (batch_size, 1024)
		new_global = (weights.unsqueeze(2) * local).sum(dim=1)
		new_global = l2norm(new_global, dim=-1)

		return new_global


class TextSA(nn.Module):
	"""
	Build global text representations by self-attention.
	Args: - local: local word embeddings, shape: (batch_size, L, 1024)
		  - raw_global: raw text by averaging words, shape: (batch_size, 1024)
	Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
	"""

	def __init__(self, embed_dim, dropout_rate):
		super(TextSA, self).__init__()

		self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
											 nn.Tanh(), nn.Dropout(dropout_rate))
		self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
											  nn.Tanh(), nn.Dropout(dropout_rate))
		self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

		self.init_weights()
		self.softmax = nn.Softmax(dim=1)

	def init_weights(self):
		for embeddings in self.children():
			for m in embeddings:
				if isinstance(m, nn.Linear):
					r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
					m.weight.data.uniform_(-r, r)
					m.bias.data.fill_(0)
				elif isinstance(m, nn.BatchNorm1d):
					m.weight.data.fill_(1)
					m.bias.data.zero_()

	def forward(self, local, raw_global):
		# compute embedding of local words and raw global text
		l_emb = self.embedding_local(local)
		g_emb = self.embedding_global(raw_global)

		# compute the normalized weights, shape: (batch_size, L)
		g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
		common = l_emb.mul(g_emb)
		weights = self.embedding_common(common).squeeze(2)
		weights = self.softmax(weights)

		# compute final text, shape: (batch_size, 1024)
		new_global = (weights.unsqueeze(2) * local).sum(dim=1)
		new_global = l2norm(new_global, dim=-1)

		return new_global


class GraphReasoning(nn.Module):
	"""
	Perform the similarity graph reasoning with a full-connected graph
	Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
	Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
	"""

	def __init__(self, sim_dim):
		super(GraphReasoning, self).__init__()

		self.graph_query_w = nn.Linear(sim_dim, sim_dim)
		self.graph_key_w = nn.Linear(sim_dim, sim_dim)
		self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
		self.relu = nn.ReLU()

		self.init_weights()

	def forward(self, sim_emb):
		sim_query = self.graph_query_w(sim_emb)
		sim_key = self.graph_key_w(sim_emb)
		sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
		sim_sgr = torch.bmm(sim_edge, sim_emb)
		sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
		return sim_sgr

	def init_weights(self):
		for m in self.children():
			if isinstance(m, nn.Linear):
				r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
				m.weight.data.uniform_(-r, r)
				m.bias.data.fill_(0)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


class AttentionFiltration(nn.Module):
	"""
	Perform the similarity Attention Filtration with a gate-based attention
	Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
	Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
	"""

	def __init__(self, sim_dim):
		super(AttentionFiltration, self).__init__()

		self.attn_sim_w = nn.Linear(sim_dim, 1)
		self.bn = nn.BatchNorm1d(1)

		self.init_weights()

	def forward(self, sim_emb):
		sim_attn = l1norm(torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1)
		sim_saf = torch.matmul(sim_attn, sim_emb)
		sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
		return sim_saf

	def init_weights(self):
		for m in self.children():
			if isinstance(m, nn.Linear):
				r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
				m.weight.data.uniform_(-r, r)
				m.bias.data.fill_(0)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


def SCAN_attention(query, context, smooth, eps=1e-8):
	"""
	query: (n_context, queryL, d)
	context: (n_context, sourceL, d)
	"""
	# --> (batch, d, queryL)
	queryT = torch.transpose(query, 1, 2)

	# (batch, sourceL, d)(batch, d, queryL)
	# --> (batch, sourceL, queryL)
	attn = torch.bmm(context, queryT)

	attn = nn.LeakyReLU(0.1)(attn)
	attn = l2norm(attn, 2)

	# --> (batch, queryL, sourceL)
	attn = torch.transpose(attn, 1, 2).contiguous()
	# --> (batch, queryL, sourceL
	attn = F.softmax(attn * smooth, dim=2)

	# --> (batch, sourceL, queryL)
	attnT = torch.transpose(attn, 1, 2).contiguous()

	# --> (batch, d, sourceL)
	contextT = torch.transpose(context, 1, 2)
	# (batch x d x sourceL)(batch x sourceL x queryL)
	# --> (batch, d, queryL)
	weightedContext = torch.bmm(contextT, attnT)
	# --> (batch, queryL, d)
	weightedContext = torch.transpose(weightedContext, 1, 2)
	weightedContext = l2norm(weightedContext, dim=-1)

	return weightedContext


# ------------------------------------------------------------

# ---------------------- CAMERA ----------------------------
class MultiViewMatching(nn.Module):
	def __init__(self, ):
		super(MultiViewMatching, self).__init__()

	def forward(self, imgs, caps, *args, **kwargs):
		# caps -- (num_caps, dim), imgs -- (num_imgs, r, dim)
		num_caps = caps.size(0)
		num_imgs, r = imgs.size()[:2]

		if num_caps == num_imgs:
			scores = torch.matmul(imgs, caps.t())  # (num_imgs, r, num_caps)
			scores = scores.max(1)[0]  # (num_imgs, num_caps)
		else:
			scores = []
			for i in range(num_caps):
				cur_cap = caps[i].unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
				cur_cap = cur_cap.expand(num_imgs, -1, -1)  # (num_imgs, 1, dim)
				cur_score = torch.matmul(cur_cap, imgs.transpose(-2, -1)).squeeze()  # (num_imgs, r)
				cur_score = cur_score.max(1, keepdim=True)[0]  # (num_imgs, 1)
				scores.append(cur_score)
			scores = torch.cat(scores, dim=1)  # (num_imgs, num_caps)

		return scores
# --------------------------------------------------------
