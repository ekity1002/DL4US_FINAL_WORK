import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout

PAD_ID = 0

class MultiheadAttention(Model):
    '''Multi-head Attention model
    model = MultiheadAttention(
        hidden_dim=512,
        head_num=8,
        dropout_rate=0.1,
    )
    model(query, memory, mask, training=True)
    '''

    def __init__(self, hidden_dim: int, head_num: int, dropout_rate: float, *args, **kwargs):
        '''コンストラクタ
        hidden_dim :隠れ層および出力の次元. head_numの倍数である必要がある
        head_num: ヘッドの数
        dropout_rate: ドロップアウトlかくりつ
        '''
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.q_dense_layer = Dense(hidden_dim, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = Dense(hidden_dim, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = Dense(hidden_dim, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = Dense(hidden_dim, use_bias=False, name='outupt_dense_layer')
        self.attention_dropout_layer = Dropout(dropout_rate)

    def call(self, 
            input: tf.Tensor,
            memory: tf.Tensor,
            attention_mask: tf.Tensor,
            training: bool
            ) -> tf.Tensor :
            '''モデルの実行
            input: query tensor
            memory: クエリに情報を与えるメモリのテンソル
            attention_mask: attention_weight に適応されるmask, 
                             shape=[batch_size, 1, q_length, k_length]
                             pad など無視する部分を True として設定すること。
            training: 学習時 or 推論時かのフラグ
            '''
            q = self.q_dense_layer(input)   #[batch_size, q_length, hidden_dim]
            k = self.k_dense_layer(memory) #[batch_size, m_length, hidden_dim]
            v = self.v_dense_layer(memory) #[batch_size, m_length, hidden_dim]

            q = self._split_head(q) #[batch_size, q_length, hidden_dim/head_num]
            k = self._split_head(k) #[batch_size, m_length, hidden_dim/head_num]
            v = self._split_head(v)

            depth = self.hidden_dim / self.head_num
            q *= depth ** -0.5 # スケールdot積用

            # 類似度 + 時系列情報のマスキング
            logit = tf.matmul(q, k, transpose_b=True) #[batch_size, head_num, q_length, k_kength]
            logit += tf.to_float(attention_mask) * input.dtype.min # mask は　pad部分などが1, 他は0

            # クエリに対応するデータの重み計算
            attention_weight = tf.nn.softmax(logit, name='attention_weight')
            attention_weight = self.attention_dropout_layer(attention_weight, training=training)

            # 重みに従って value から情報を取り出す
            attention_output = tf.matmul(attention_weight, v) #[batch_size, head_num, q_length, hidden_dim/head_dim]
            attention_output = self._combine_head(attention_output) #[batch_size, q_length, hidden_dim]
            return self.output_dense_layer(attention_output)


    def _split_head(self, x: tf.Tensor) -> tf.Tensor:
        '''
        入力の tensor の hidden_dim の次元をいくつかの head に分割する
        input_shape = [batch_size, length, hidden_dim] の時
        output_shape = [batch_size, length, hidden_dim//head_num] となる
        '''
        with tf.name_scope('split_head'):
            batch_size, length, hidden_dim = tf.unstack(tf.shape(x))
            x = tf.reshape(x, [batch_size, length, self.head_num, self.hidden_dim // self.head_num])
            return tf.transpose(x, [0, 2, 1, 3])

    def _combine_head(self, x: tf.Tensor) -> tf.Tensor:
        '''入力の Tensor のヘッドを結合する _split_head の逆変換
        input_shape: [batch_size, head_num, length, hidden_dim//head_num] の時
        output_shape: [batch_size, length, hidden_dim]
        '''
        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0,2,1,3])
            return tf.reshape(x, [batch_size, length, self.hidden_dim])
        
class SelfAttention(MultiheadAttention):
    def call(self,
             input: tf.Tensor,
             attention_mask: tf.Tensor,
             training: bool,
             ) -> tf.Tensor:
             return super().call(
                 input=input,
                 memory=input,
                 attention_mask = attention_mask,
                 training=training
             )
        
class FeedForwardNetwork(Model):
    '''Transformer 用の Position-wise Feedforward Neural Network です'''
    def __init__(self, hidden_dim: int, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.filter_dense_layer = Dense(hidden_dim*4, use_bias=True,
                                        activatiion=tf.nn.relu, name='filter_layer')
        self.output_dense_layer = Dense(hidden_dim, use_bias=True, name='output_layer')
        self.dropout_layer = Dropout(dropout_rate)

    def call(self, input: tf.Tensor, training: bool) -> tf.Tensor:
        '''FeedForwadNetwork 適用する
        input: shape = [batch_size, length, hidden_dim]
        return: shape= [batch_size, length, hidden_dim]
        '''
        tensor = self.filter_dense_layer(input)
        tensor = self.dropout_layer(tensor, training=training)
        return self.output_dense_layer(tensor)
    
class LayerNormalization(tf.keras.layers.Layer):
    def build(self, input_shape: tf.TensorShape) -> None:
        hidden_dim = input_shape[-1]
        self.scale = self.add_weight('layer_norm_scale', shape=[hidden_dim],
                                     initializer=tf.ones_initializer())
        self.bias = self.add_weight('layer_norm_bias', [hidden_dim],
                                    initializer=tf.zeros_initializer())
        super().build(input_shape)

    def call(self, x: tf.Tensor, epsilon: float =1e-6) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x-mean), axis=[-1], keepdims=True)
        norm_x = (x-mean) * tf.rsqrt(variance + epsilon)

        return norm_x * self.scale + self.bias
    
class ResidualNormalizationWrapper(Model):
    def __init__(self, layer: tf.keras.layers.Layer, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.layer_normalization = LayerNormalization()
        self.dropout_layer = Dropout(dropout_rate)

    def call(self, input: tf.Tensor, training: bool, *args, **kwargs) -> tf.Tensor:
        tensor = self.layer_normalization(input)
        tensor = self.layer(tensor, training=training, *args, **kwargs)
        tensor = self.dropout_layer(tensor, training=training)
        return input + tensor
    
class AddPositionalEncoding(tf.keras.layers.Layer):
    '''入力テンソルに対し位置情報を付与して返すレイヤー
    PE_{pos, 2i} = sin(pos / 10000^{2i / d_model})
    PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
    '''
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        fl_type = inputs.dtype
        batch_size, max_length, depth = tf.unstack(tf.shape(inputs))

        depth_counter = tf.range(depth) // 2 * 2 #0, 0, 2, 2, 4, ...
        # tf.expand : ２つめの引数でしてした番号の shape 位置に次元追加
        # tf.tile: 第一引数の tensor を 第二引数 で指定した形に並べる
        depth_matrix = tf.tile(tf.expand_dims(depth_counter, 0), [max_length, 1]) #[max_length, depth]
        depth_matrix = tf.pow(10000.0, tf.cast(depth_matrix / depth, fl_type)) #[max_length, depth]

        # cos(x) == sin(x+ pi/2)
        phase = tf.cast(tf.range(depth) % 2, fl_type) * math.pi / 2 #0, pi/2, 0, pi/2, ...
        phase_matrix = tf.tile(tf.expand_dims(phase, 0), [max_length, 1]) #[max_length, depth]

        pos_counter = tf.range(max_length)
        pos_matrix = tf.cast(tf.tile(tf.expand_dims(pos_counter, 1), [1, depth]), fl_type) #[max_length, depth]

        positional_encoding = tf.sin(pos_matrix / depth_matrix + phase_matrix)

        # [batch_size, max_length, depth]
        positional_encoding = tf.tile(tf.expand_dims(positional_encoding, 0), [batch_size, 1, 1])
        return inputs + positional_encoding


class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, voca_size: int, embedding_dim: int, dtype=tf.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dtype_ = dtype

    def build(self, input_shape: tf.TensorShape) -> None:
        self.lookup_table = self.add_variable(
            name='token_embedding',
            shape=[self.vocab_size, self.embedding_dim],
            dtype=self.dtype,
            initializer=tf.random_normal_initializer(0., self.embedding_dim ** -0.5)
        )
        super().build(input_shape)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        mask = tf.to_float(tf.not_equal(input, PAD_ID))
        embedding = tf.nn.embedding_lookup(self.lookup_table, input)
        embedding *= tf.exmapd_dims(mask, -1) # もともとpadだった部分を 0 にする
        return embedding * self.embedding_dim ** 0.5
    
    
class Encoder(Model):
    '''トークン列をベクトル列にエンコードする Encoder'''
    def __init__(
        self,
        vocab_size: int,
        hopping_num: int,
        head_num: int,
        hidden_dim: int,
        dropout_rate: float,
        max_length: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = Dropout(dropout_rate)

        self.attention_block_list: List[List[Model]] = []
        # hopping_Num だけ SelfAttention, FeedForward の残差構造を繰り返す
        for _ in range(hopping_num):
            attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper(atteition_lauer, dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),
            ])

        self.output_normalization = LayerNormalization()

    def call(
        self,
        input: tf.Tensor,
        self_attention_mask: tf.Tensor,
        training: bool,
        ) -> tf.Tensor:
        '''モデル実行
        input_shape = [batch_size, length]
        training: 学習時は True
        return: shape = [batch_size, length, hidden_dim]
        '''
        #[batch_size, length, hidden_dim]
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)

        for i, layers in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query = attention_layer(query, attention_mask=self_attention_mask, training=training)
                query = ffn_layer(query, training=training)
            #[batch_size, length, hidden_dim]
        return self.output_normalization(query)
    
class Decoder(Model):
    def __init__(
        self,
        vocab_size: int,
        hopping_num: int,
        head_num: int,
        hidden_dim: int,
        dropout_rate: float,
        max_length: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dimk = hidden_dim
        self.dropout_rate = dropout_rate

        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = Dropout(dropout_rate)

        self.attention_block_list: List[List[Model]] = []
        for _ in range(hopping_num):
            self_attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
            enc_dec_attention_layer = MultiheadAttention(hidden_dim, head_num, dropout, name='enc_dec_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, head_num, dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper(self_attention_layer, dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper(enc_dec_attention_layer, dropout_rate, name='enc_dec_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),                                              
            ])

        self.output_normalization = LayerNormalization()

        # 本家ではここは　TokenEmbedding の重みを転地したものを使っている
        self.output_dense_layer = Dense(vocab_size, use_bias=False)

    def call(
        self,
        input: tf.Tensor,
        encoder_output: tf.Tensor,
        self_attention_mask: tf.Tensor,
        enc_dec_attention_mask: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        '''
        input: shape=[batch_size, length]
        traingin: 学習時 True
        return: shape = [batch_size, length, hidden_dim]
        '''
        # [batch_size, length, hidden_dim]
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)

        for i, layers in enumerate(self.attention_block_list):
            self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query = self_attention_layer(query, attention_mask=self_attention_mask, training=training)
                query = enc_dec_attention_layer(query, memory=encoder_output,
                                                attention_mask=enc_dec_attention_mask, training=training)
                query = ffn_layer(query, training=training)
        query = self.output_normalization(query) #[batch_size, lengt, hidden_dim]
        return self.output_dense_layer(query) #[batch_size, length, vocab_size]
    
    
class Transformer(Model):
    def __init__(
        self,
        vocab_size: int,
        hopping_num: int = 4,
        head_num: int = 8,
        hidden_dim: int = 512,
        dropout_rate: float = 0.1,
        max_length: int = 200,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.max_length = max_length

        self.encoder = Encoder(
            vocab_size = vocab_size,
            hopping_num=hopping_num,
            head_num=head_num,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            max_length=max_length,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hopping_num=hopping_num,
            head_num=head_num,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            max_length=max_length,
        )

    def build_graph(self, name='transformer') -> None:
        '''
        学習、推論のためのグラフを構築
        '''
        with tf.name_scope(name):
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            # [batch_size, max_length]
            self.encoder_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder_input')
            # [batch_size]
            self.decoder_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='decoder_input')

            logit = self.call(
                encoder_input=self.encoder_input,
                decoder_input=self.decoder_input[:, :-1], #入力は EOSを含めない
                training=self.is_training,
            )
            decoder_target = self.decoder_input[:, 1:] #出力は BOS　を含めない

            self.prediction = tf.nn.softmax(logit, name='prediction')

            with tf.name_scope('metrics'):
                xentropy, weights = padded_cross_entropy_loss(
                    logit, decoder_target, smoothing=0.05, vocab_size=self.vocab_size,
                )
                self.loss = tf.identity(tf.reduce_sum(xentropy)/ tf.reduce_sum(weights), name='loss')

                accuracies, weights = padded_accuracy(logit, decoder_target)
                self.acc = tf.identity(tf.reduce_sum(accuracies) / tf.reduce_sum(weights), name='acc')

    def call(self, encoder_input: tf.Tensor, decoder_input: tf.Tensor, training: bool) -> tf.Tensor:
        enc_attention_mask = self._create_enc_attention_mask(encoder_input)
        dec_self_attention_mask = self._create_dec_self_attention_mask(decoder_input)

        encoder_output = self.encoder(
            encoder_input,
            self_attention_mask=enc_attention_mask,
            training=training,
        )
        decoder_output = self.decoder(
            decoder_input,
            encoder_output,
            self_attention_mask=dec_self_attention_mask,
            enc_dec_attention_mask=enc_attention_mask,
            training=training,
        )
        return decoder_output

    def _create_enc_attention_mask(self, encoder_input: tf.Tensor):
        with tf.name_scope('enc_attention_mask'):
            batch_size, length = tf.unstack(tf.shape(encoder_input))
            pad_array = tf.equal(encoder_input, PAD_ID) #[batch_size, m_length]
            # shape broadcasting で　[batch_size, head_num, (m|q)length, m_llength] になる
            return tf.reshape(pad_array, [batch_size, 1, 1, length])

    def _create_dec_self_attention_mask(self, decoder_input: tf.Tensor):
        with tf.name_scope('dec_self_attention_mask'):
            batch_size, length = tf.unstack(tf.shape(decoder_input))
            pad_array = tf.equal(decoder_input, PAD_ID) #[batch_size, m_length]
            pad_array = tf.reshape(pad_array, [batch_size, 1, 1, length])

            autoregression_array = tf.logical_not(
                tf.matrix_band_part(tf.ones([length, length], dtype=tf.bool), -1, 0)) #した三角 が False
            autoregression_array = tf.reshape(autoregression_array, [1, 1, length, length])
            return tf.logical_or(pad_array, autoregression_array)