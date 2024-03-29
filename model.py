import tensorflow as tf
import tensorflow.contrib.slim as slim


class Hirar(object):
    '''
    Hierarchical Features for caricature geneation
    '''

    def __init__(self, mode='train', learning_rate=0.0003,
                 n_classes=10, class_weight=1.0, feat_layer=5,
                 skip=True, skip_layers=2, adv_weight=1.0,
                 tv_weight=1.0):

        self.mode = mode
        self.skip_layers = skip_layers
        self.skip = skip
        self.tv_weight = tv_weight
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.class_weight = class_weight
        self.adv_weight = adv_weight
        self.feat_layer = feat_layer
        self.depth_dict = {1: 64,
                           2: 128,
                           3: 256,
                           4: 512,
                           5: 512}

    def classifier(self, encodings, scope_suffix='caric', reuse=False):

        assert scope_suffix in ['caric', 'real']
        with tf.variable_scope('classifier_' + scope_suffix,
                               reuse=reuse):
            flattened = slim.flatten(encodings)
            l1 = slim.fully_connected(flattened, 400, scope='fc1')
            l2 = slim.fully_connected(l1, self.n_classes, scope='fc2')
            return l2

    def encoder(self, images, reuse=False, scope_suffix='caric'):

        assert scope_suffix in ['caric', 'real']
        features = []

        # images: (batch, 64, 64, 3) or (batch, 64, 64, 1)
        if images.get_shape()[3] == 1:
            # Replicate the gray scale image 3 times.
            images = tf.image.grayscale_to_rgb(images)

        with tf.variable_scope('encoder_' + scope_suffix, reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                activation_fn=None,
                                stride=2,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95,
                                    center=True, scale=True,
                                    activation_fn=tf.nn.relu,
                                    is_training=self.mode in ['train',
                                                              'pretrain']):
                    # (batch_size, 32, 32, 64)
                    net = slim.conv2d(images, 64, [3, 3],
                                      scope='conv1')
                    lr1 = slim.batch_norm(net, scope='bn1')
                    features.append(lr1)
                    # (batch_size, 16, 16, 128)
                    net = slim.conv2d(lr1, 128, [3, 3],
                                      scope='conv2')
                    lr2 = slim.batch_norm(net, scope='bn2')
                    features.append(lr2)
                    # (batch_size, 8, 8, 256)
                    net = slim.conv2d(lr2, 256, [3, 3],
                                      scope='conv3')
                    lr3 = slim.batch_norm(net, scope='bn3')
                    features.append(lr3)
                    # (batch_size, 4, 4, 512)
                    net = slim.conv2d(lr3, 512, [3, 3],
                                      scope='conv4')
                    lr4 = slim.batch_norm(net, scope='bn4')
                    features.append(lr4)
                    # (batch_size, 1, 1, 512)
                    net = slim.conv2d(lr4, 512, [4, 4], padding='VALID',
                                      scope='conv5')
                    lr5 = slim.batch_norm(net, activation_fn=tf.nn.tanh,
                                          scope='bn5')
                    features.append(lr5)

                    # (batch_size, n_classes)
                    logits = self.classifier(encodings=lr5,
                                             scope_suffix=scope_suffix,
                                             reuse=reuse)
                    return features, logits

    def decoder(self, inputs, reuse=False,
                layer=5, scope_suffix='caric'):

        assert layer in [1, 2, 3, 4, 5]
        # inputs: (batch, 1, 1, 512)
        net = inputs
        with tf.variable_scope('decoder_' + scope_suffix, reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose],
                                padding='SAME', activation_fn=None,
                                stride=2,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm],
                                    decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.relu,
                                    is_training=(self.mode == 'train')):

                    # (batch, 1, 1, 512) -> (batch_size, 4, 4, 512)
                    if layer == 5:
                        net = slim.conv2d_transpose(
                            net, 512, [4, 4],
                            padding='VALID', scope='conv_transpose1')
                        net = slim.batch_norm(net, scope='bn1')
                    # (batch_size, 4, 4, 512) -> (batch_size, 8, 8, 256)
                    if layer >= 4:
                        net = slim.conv2d_transpose(net, 256, [3, 3],
                                                    scope='conv_transpose2')
                        net = slim.batch_norm(net, scope='bn2')
                    # (batch_size, 8, 8, 256) -> (batch_size, 16, 16, 128)
                    if layer >= 3:
                        net = slim.conv2d_transpose(net, 128, [3, 3],
                                                    scope='conv_transpose3')
                        net = slim.batch_norm(net, scope='bn3')
                    # (batch_size, 16, 16, 128) -> (batch_size, 32, 32, 64)
                    if layer >= 2:
                        net = slim.conv2d_transpose(net, 64, [3, 3],
                                                    scope='conv_transpose4')
                        net = slim.batch_norm(net, scope='bn4')
                    # (batch_size, 32, 32, 64) -> (batch_size, 64, 64, 3)
                    net = slim.conv2d_transpose(net, 3, [3, 3],
                                                activation_fn=tf.nn.tanh,
                                                scope='conv_transpose5')
                    return net

    def discriminator(self, features, layer=5, reuse=False):

        # images: (batch, 64, 64, 3)
        def lrelu(x, leak=0.2, name='leaky_relu'):
            return tf.maximum(x, leak * x)

        net = features
        assert layer in [0, 1, 2, 3, 4, 5]
        with tf.variable_scope('discriminator_layer_%d' % layer, reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                activation_fn=None,
                                stride=2,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95,
                                    center=True, scale=True,
                                    activation_fn=lrelu,
                                    is_training=(self.mode == 'train')):

                    # (batch, 64, 64, 3) -> (batch_size, 32, 32, 64)
                    if layer == 0:
                        net = slim.conv2d(net, 64, [3, 3],
                                          scope='conv1')
                        net = slim.batch_norm(net, scope='bn1')
                    # (batch_size, 32, 32, 64) -> (batch_size, 16, 16, 128)
                    if layer <= 1:
                        net = slim.conv2d(net, 128, [3, 3], scope='conv2')
                        net = slim.batch_norm(net, scope='bn2')
                    # (batch_size, 16, 16, 128) -> (batch_size, 8, 8, 256)
                    if layer <= 2:
                        net = slim.conv2d(net, 256, [3, 3], scope='conv3')
                        net = slim.batch_norm(net, scope='bn3')
                    # (batch_size, 8, 8, 256) -> (batch_size, 4, 4, 512)
                    if layer <= 3:
                        net = slim.conv2d(net, 512, [3, 3], scope='conv4')
                        net = slim.batch_norm(net, scope='bn4')
                    # (batch_size, 4, 4, 512) -> (batch_size, 1, 1, 512)
                    # if layer <= 4:
                        # net = slim.conv2d(net, 512, [4, 4], padding='VALID',
                        # scope='conv5')
                        # net = slim.batch_norm(net, scope='bn5')
                    # net = slim.flatten(net)
                    # (batch_size, 512) -> #(batch_size, 50)
                    # net = slim.fully_connected(net, 50, scope='fc6')
                    # (batch_size, 50) -> #(batch_size, )
                    # net = slim.fully_connected(net, 1, scope='fc7')
                    return net

    def transformer(self, features, layer=5, reuse=False):

        def lrelu(x, leak=0.2, name='leaky_relu'):
            return tf.maximum(x, leak * x)

        assert layer in [1, 2, 3, 4, 5]
        scope = 'transformer_layer_%d' % layer
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                activation_fn=None,
                                stride=1,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95,
                                    center=True, scale=True,
                                    activation_fn=lrelu,
                                    is_training=(self.mode == 'train')):

                    if layer == 5:
                        stride = 1
                    else:
                        stride = 3

                    depth = self.depth_dict[layer]
                    net = slim.conv2d(features, depth,
                                      [stride, stride],
                                      scope='conv1')
                    net = slim.batch_norm(net, scope='bn1')

                    for i in xrange(self.skip_layers - 1):
                        net = slim.conv2d(net, depth,
                                          [stride, stride],
                                          scope='conv%d' % (i + 2))
                        net = slim.batch_norm(net,
                                              scope='bn%d' % (i + 2))
                        if self.skip and (i + 1) % 2 == 0:
                            net = net + features

                    return net

    def build_model(self):

        if self.mode == 'pretrain':
            self.real_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                              'real_faces')
            self.caric_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                               'caric_faces')
            self.real_labels = tf.placeholder(tf.int64, [None],
                                              'real_labels')
            self.caric_labels = tf.placeholder(tf.int64, [None],
                                               'caric_labels')

            # logits and accuracy
            self.real_enc, self.real_logits = self.encoder(self.real_images,
                                                           scope_suffix='real')
            self.caric_enc, self.caric_logits = self.encoder(self.caric_images,
                                                             scope_suffix='caric')

            self.reconst_carics, self.reconst_reals = [], []
            reuse = False
            for layer, feature in zip(xrange(5, 0, -1),
                                      reversed(self.real_enc)):
                self.reconst_reals.insert(0, self.decoder(feature,
                                                          layer=layer,
                                                          scope_suffix='real',
                                                          reuse=reuse))
                reuse = True
            reuse = False
            for layer, feature in zip(xrange(5, 0, -1),
                                      reversed(self.caric_enc)):
                self.reconst_carics.insert(0, self.decoder(feature,
                                                           layer=layer,
                                                           scope_suffix='caric',
                                                           reuse=reuse))
                reuse = True

            _, self.fake_caric_logits = self.encoder(self.reconst_carics[-1],
                                                     scope_suffix='caric',
                                                     reuse=True)
            _, self.fake_real_logits = self.encoder(self.reconst_reals[-1],
                                                    scope_suffix='real',
                                                    reuse=True)
            self.labels = tf.concat([self.real_labels, self.caric_labels], 0)
            self.logits = tf.concat([self.real_logits, self.caric_logits], 0)
            self.fake_logits = tf.concat([self.fake_real_logits,
                                          self.fake_caric_logits], 0)

            self.pred = tf.argmax(self.logits, 1)
            self.fake_pred = tf.argmax(self.fake_logits, 1)
            self.correct_pred = tf.equal(self.pred,
                                         self.labels)
            self.correct_fake_pred = tf.equal(self.fake_pred,
                                              self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred,
                                                   tf.float32))
            self.fake_accuracy = tf.reduce_mean(tf.cast(
                self.correct_fake_pred, tf.float32))

            # loss and train op
            self.loss_reconst_caric, self.loss_reconst_real = 0.0, 0.0
            for reconst_real, wt in zip(self.reconst_reals, range(1, 6)):
                self.loss_reconst_real += tf.reduce_mean(tf.square(
                    self.real_images - reconst_real)) * wt
            for reconst_caric, wt in zip(self.reconst_carics, range(1, 6)):
                self.loss_reconst_caric += tf.reduce_mean(tf.square(
                    self.caric_images - reconst_caric)) * wt
            self.loss_reconst = self.loss_reconst_caric \
                + self.loss_reconst_real

            self.loss_class = \
                tf.losses.sparse_softmax_cross_entropy(self.labels,
                                                       self.logits) \
                + tf.losses.sparse_softmax_cross_entropy(self.labels,
                                                         self.fake_logits)

            self.loss = self.loss_class * self.class_weight \
                + self.loss_reconst
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = slim.learning.create_train_op(self.loss,
                                                          self.optimizer,
                                                          clip_gradient_norm=1)

            # summary op
            loss_reconst_summary = tf.summary.scalar('reconstruction_loss',
                                                     self.loss_reconst)
            loss_reconst_real_summ = tf.summary.scalar('reconstruction_real',
                                                       self.loss_reconst_real)
            loss_reconst_caric_summ = tf.summary.scalar('reconstruction_caric',
                                                        self.loss_reconst_caric)
            loss_class_summary = tf.summary.scalar('classification_loss',
                                                   self.loss_class)
            loss_summary = tf.summary.scalar('combined loss',
                                             self.loss)
            accuracy_summary = tf.summary.scalar('accuracy',
                                                 self.accuracy)
            fake_accuracy_summary = tf.summary.scalar('fake_accuracy',
                                                      self.fake_accuracy)

            reconst_real_2 = tf.summary.image('reconst_real_2',
                                              self.reconst_reals[1])
            reconst_real_5 = tf.summary.image('reconst_real_5',
                                              self.reconst_reals[4])
            reconst_caric_2 = tf.summary.image('reconst_caric_2',
                                               self.reconst_carics[1])
            reconst_caric_5 = tf.summary.image('reconst_caric_5',
                                               self.reconst_carics[4])
            caric_image_summary = tf.summary.image('caric_images',
                                                   self.caric_images)
            self.summary_op = tf.summary.merge([loss_summary,
                                                loss_reconst_summary,
                                                loss_reconst_caric_summ,
                                                loss_reconst_real_summ,
                                                loss_class_summary,
                                                reconst_caric_2,
                                                reconst_caric_5,
                                                reconst_real_2,
                                                reconst_real_5,
                                                caric_image_summary,
                                                accuracy_summary,
                                                fake_accuracy_summary])

        elif self.mode == 'eval':
            self.images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                         'real_faces')

            # source domain
            self.fx = self.content_extractor(self.images)
            self.sampled_images = self.generator(self.fx)

        elif self.mode == 'train':
            self.real_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                              'real_faces')
            self.caric_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                               'caric_faces')
            self.real_labels = tf.placeholder(tf.int64, [None],
                                              'real_labels')
            self.caric_labels = tf.placeholder(tf.int64, [None],
                                               'caric_labels')

            # encodings, transformations and reconstructions
            self.real_enc, _ = self.encoder(self.real_images,
                                            scope_suffix='real')
            self.caric_enc, self.caric_logits = self.encoder(self.caric_images,
                                                             scope_suffix='caric')

            self.reconst_caric = self.decoder(inputs=self.caric_enc[self.feat_layer - 1],
                                              layer=self.feat_layer,
                                              scope_suffix='caric')
            self.reconst_real = self.decoder(inputs=self.real_enc[self.feat_layer - 1],
                                             layer=self.feat_layer,
                                             scope_suffix='real')
            self.trans_real_feat = self.transformer(features=self.real_enc[self.feat_layer - 1],
                                                    layer=self.feat_layer)
            self.reconst_trans = self.decoder(inputs=self.trans_real_feat,
                                              reuse=True,
                                              layer=self.feat_layer,
                                              scope_suffix='caric')
            _, self.reconst_logits = self.encoder(self.reconst_trans,
                                                  scope_suffix='caric',
                                                  reuse=True)

            # discriminator scores
            self.pos_score = self.discriminator(features=self.caric_enc[self.feat_layer - 1],
                                                layer=self.feat_layer)
            self.neg_score = self.discriminator(features=self.trans_real_feat,
                                                layer=self.feat_layer,
                                                reuse=True)
            self.real_prob = tf.nn.sigmoid(self.pos_score)
            self.fake_prob = tf.nn.sigmoid(self.neg_score)

            self.reconst_score = self.discriminator(features=self.reconst_trans,
                                                    layer=0)
            self.caric_score = self.discriminator(features=self.caric_images,
                                                  layer=0,
                                                  reuse=True)
            self.caric_prob = tf.nn.sigmoid(self.caric_score)
            self.reconst_prob = tf.nn.sigmoid(self.reconst_score)

            # accuracy
            pred = tf.argmax(self.reconst_logits, 1)
            correct_pred = tf.equal(pred,
                                    self.real_labels)
            self.trans_accr = tf.reduce_mean(tf.cast(
                correct_pred, tf.float32))

            pred = tf.argmax(self.caric_logits, 1)
            correct_pred = tf.equal(pred,
                                    self.caric_labels)
            self.caric_accr = tf.reduce_mean(tf.cast(
                correct_pred, tf.float32))

            # loss_decoder
            def get_reconst_loss(a, b):
                return tf.reduce_mean(tf.losses.absolute_difference(a, b))

            self.loss_decoder = get_reconst_loss(self.reconst_caric,
                                                 self.caric_images) \
                + get_reconst_loss(self.reconst_real,
                                   self.real_images)

            # classification_loss
            self.loss_class = \
                tf.losses.sparse_softmax_cross_entropy(self.real_labels,
                                                       self.reconst_logits) \
                + tf.losses.sparse_softmax_cross_entropy(self.caric_labels,
                                                         self.caric_logits)
            # adversarial_loss
            EPS = 1e-32
            self.loss_disc = -tf.reduce_mean(
                tf.log(self.real_prob + EPS)
                + tf.log(1 - self.fake_prob + EPS)
                + tf.log(self.caric_prob + EPS) * 10.0
                + tf.log(1 - self.reconst_prob + EPS) * 10.0) \
                # - tf.reduce_mean(self.pos_score - self.neg_score +
            # 10.0 * (self.caric_score
            # - self.reconst_score)) * 1e-5

            self.loss_gen = - tf.reduce_mean(
                tf.log(self.fake_prob)
                + tf.log(self.reconst_prob) * 10.0) \
                # - tf.reduce_mean(
            # 1e-5 * (self.neg_score + self.reconst_score * 10.0))

            self.loss_tv = self.tv_weight * tf.reduce_mean(
                tf.image.total_variation(images=self.reconst_trans))

            # transformer_loss
            self.loss_transformer = self.loss_gen * self.adv_weight \
                + self.loss_class * self.class_weight \
                + self.loss_tv

            # optimizer
            self.enc_opt = tf.train.RMSPropOptimizer(self.learning_rate)
            self.dec_opt = tf.train.RMSPropOptimizer(self.learning_rate)
            self.disc_opt = tf.train.RMSPropOptimizer(self.learning_rate)
            self.trans_opt = tf.train.RMSPropOptimizer(self.learning_rate)

            # model variables
            all_vars = tf.trainable_variables()
            disc_vars = \
                [var for var in all_vars if 'discriminator' in var.name]
            trans_vars = \
                [var for var in all_vars if 'transformer' in var.name]
            dec_vars = \
                [var for var in all_vars if 'decoder_' in var.name]
            # enc_vars = \
                # [var for var in all_vars if (
                    # 'encoder_' in var.name or 'classifier_' in var.name)]

            # train op
            with tf.variable_scope('train_op', reuse=False):
                self.disc_op = slim.learning.create_train_op(
                    self.loss_disc,
                    self.disc_opt,
                    variables_to_train=disc_vars,
                    clip_gradient_norm=0.01)
                self.trans_op = slim.learning.create_train_op(
                    self.loss_transformer,
                    self.trans_opt,
                    variables_to_train=trans_vars)
                self.dec_op = slim.learning.create_train_op(
                    self.loss_decoder,
                    self.dec_opt,
                    variables_to_train=dec_vars)
                # self.enc_op = slim.learning.create_train_op(
                    # self.loss_class,
                    # self.enc_opt,
                    # variables_to_train=enc_vars)

            # summary op
            gen_loss_summary = tf.summary.scalar('gen_loss',
                                                 self.loss_gen)
            tv_loss_summary = tf.summary.scalar('loss_total_variation',
                                                 self.loss_tv)
            class_loss_summary = tf.summary.scalar('loss_class',
                                                   self.loss_class)
            dec_loss_summary = tf.summary.scalar('loss_dec',
                                                 self.loss_decoder)
            t_accuracy_summary = tf.summary.scalar('trans_accr',
                                                   self.trans_accr)
            c_accuracy_summary = tf.summary.scalar('caric_accr',
                                                   self.caric_accr)
            disc_loss_summary = tf.summary.scalar('disc_loss',
                                                  self.loss_disc)
            caric_prob_summary = tf.summary.scalar('caric_prob',
                                                  tf.reduce_mean(self.caric_prob))
            reconst_prob_summary = tf.summary.scalar('reconst_prob',
                                                  tf.reduce_mean(self.reconst_prob))
            real_prob_summary = tf.summary.scalar('real_prob',
                                                  tf.reduce_mean(self.real_prob))
            fake_prob_summary = tf.summary.scalar('fake_prob',
                                                  tf.reduce_mean(self.fake_prob))
            trans_loss_summary = tf.summary.scalar('transformer_loss',
                                                   self.loss_transformer)
            real_images_summary = tf.summary.image('real_images',
                                                   self.real_images)
            caric_images_summary = tf.summary.image('caric_images',
                                                    self.caric_images)
            real_reconst_summary = tf.summary.image('real_reconst',
                                                    self.reconst_real)
            trans_reconst_summary = tf.summary.image('reconst_trans',
                                                     self.reconst_trans)
            caric_reconst_summary = tf.summary.image('caric_reconst',
                                                     self.reconst_caric)
            self.summary_op = tf.summary.merge([gen_loss_summary,
                                                dec_loss_summary,
                                                t_accuracy_summary,
                                                c_accuracy_summary,
                                                class_loss_summary,
                                                real_prob_summary,
                                                fake_prob_summary,
                                                tv_loss_summary,
                                                disc_loss_summary,
                                                caric_prob_summary,
                                                reconst_prob_summary,
                                                trans_loss_summary,
                                                real_images_summary,
                                                caric_images_summary,
                                                real_reconst_summary,
                                                trans_reconst_summary,
                                                caric_reconst_summary])
