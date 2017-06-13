import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
from data_loader import DataLoader
from random import sample


class Solver(object):

    def __init__(self, model, batch_size=100, pretrain_iter=20000,
                 train_iter=2000, sample_iter=100,
                 real_dir='real-face', caric_dir='caricature-face',
                 combined_dir='class-combined.pkl',
                 log_dir='logs', n_classes=200,
                 sample_save_path='sample',
                 model_save_path='model',
                 pretrained_model='model/pre_model-4000',
                 test_model='model/dtn_ext-400',
                 src_disc_rep=1,
                 src_gen_rep=1,
                 trg_disc_rep=1,
                 trg_gen_rep=1):

        self.loader = DataLoader(batch_size)
        self.model = model
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.pretrain_iter = pretrain_iter
        self.combined_dir = combined_dir
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.real_dir = real_dir
        self.caric_dir = caric_dir
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.src_disc_rep = src_disc_rep
        self.src_gen_rep = src_gen_rep
        self.trg_disc_rep = trg_disc_rep
        self.trg_gen_rep = trg_gen_rep

    def load_real(self, image_dir, split='train'):
        print ('loading real faces..')
        image_file = 'train.pkl' if split == 'train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            real_faces = pickle.load(f)
        mean = np.mean(real_faces['X'])
        images = real_faces['X'] / mean - 1
        labels = real_faces['y']
        print ('finished loading real faces..!')
        return images, labels

    def load_caric(self, image_dir, split='train'):
        print ('loading caricature faces..')
        image_file = 'train.pkl' if split == 'train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            caric = pickle.load(f)
        mean = np.mean(caric['X'])
        images = caric['X'] / mean - 1
        labels = caric['y']
        print ('finished loading caricature faces..!')
        return images, labels

    def load_combined(self):
        print('loading combined images ...')
        with open(self.combined_dir, 'rb') as f:
            combined_imgs = pickle.load(f)
        for lbl in combined_imgs:
            mean_r = np.mean(combined_imgs[lbl]['real'])
            mean_c = np.mean(combined_imgs[lbl]['caric'])
            combined_imgs[lbl]['real'] = combined_imgs[lbl]['real'] / mean_r - 1
            combined_imgs[lbl]['caric'] = combined_imgs[lbl]['caric'] / mean_c - 1
        print('finished loading combined_imgs')
        return combined_imgs

    def merge_images(self, sources, targets, k=10):
        _, h, w, _ = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([row * h, row * w * 2, 3])

        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h, :] = s
            merged[i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h, :] = t
        return merged

    def get_pairs(self, combined_images, label_set, set_type='positive'):
        def get_pos_pair(label):
            toss = np.random.uniform()
            if toss < 0.5:
                real_img = sample(combined_images[label]['real'], 1)[0]
                caric_img = sample(combined_images[label]['caric'], 1)[0]
                return [real_img, caric_img]
            elif toss > 0.75:
                try:
                    return sample(combined_images[label]['caric'], 2)
                except:
                    return sample(combined_images[label]['real'], 2)
            else:
                try:
                    return sample(combined_images[label]['real'], 2)
                except:
                    real_img = sample(combined_images[label]['real'], 1)[0]
                    caric_img = sample(combined_images[label]['caric'], 1)[0]
                    return [real_img, caric_img]

        def get_neg_pair(label):
            toss = np.random.uniform()
            neg_lbl = sample(label_set - set([label]), 1)[0]
            if toss < 0.5:
                neg_img = sample(combined_images[neg_lbl]['real'], 1)[0]
                img = sample(combined_images[label]['caric'], 1)[0]
            elif toss > 0.75:
                neg_img = sample(combined_images[neg_lbl]['caric'], 1)[0]
                img = sample(combined_images[label]['caric'], 1)[0]
            else:
                neg_img = sample(combined_images[neg_lbl]['caric'], 1)[0]
                img = sample(combined_images[label]['caric'], 1)[0]
            return [img, neg_img]

        # some labels for positive pairs
        perm = sample(label_set, self.batch_size)
        if set_type == 'positive':
            pos_ones, pos_twos = zip(*map(get_pos_pair, perm))
            return np.asarray(pos_ones), np.asarray(pos_twos)
        else:
            neg_ones, neg_twos = zip(*map(get_neg_pair, perm))
            return np.asarray(neg_ones), np.asarray(neg_twos)

    def pretrain(self):

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        # load real faces
        train_images_r, train_labels_r = self.load_real(self.real_dir,
                                                        split='train')
        test_images_r, test_labels_r = self.load_real(self.real_dir,
                                                      split='test')
        train_images_c, train_labels_c = self.load_caric(self.caric_dir,
                                                         split='train')
        test_images_c, test_labels_c = self.load_caric(self.caric_dir,
                                                       split='test')

        self.loader.add_dataset('caric_faces_tr', train_images_c)
        self.loader.add_dataset('caric_labels_tr', train_labels_c)
        self.loader.add_dataset('real_faces_tr', train_images_r)
        self.loader.add_dataset('real_labels_tr', train_labels_r)
        self.loader.link_datasets('caric_tr',
                                  ['caric_labels_tr', 'caric_faces_tr'])
        self.loader.link_datasets('real_tr',
                                  ['real_labels_tr', 'real_faces_tr'])
        self.loader.add_dataset('caric_faces_te', test_images_c)
        self.loader.add_dataset('caric_labels_te', test_labels_c)
        self.loader.add_dataset('real_faces_te', test_images_r)
        self.loader.add_dataset('real_labels_te', test_labels_r)
        self.loader.link_datasets('caric_te',
                                  ['caric_labels_te', 'caric_faces_te'])
        self.loader.link_datasets('real_te',
                                  ['real_labels_te', 'real_faces_te'])

        # build a graph
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            if self.pretrained_model != '':
                print ('loading pretrained model ..')
                saver.restore(sess, self.pretrained_model)
            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())

            for step in range(self.pretrain_iter + 1):
                caric_labels, caric_images = \
                    self.loader.next_group_batch('caric_tr')
                real_labels, real_images = \
                    self.loader.next_group_batch('real_tr')
                feed_dict = {model.real_images: real_images,
                             model.real_labels: real_labels,
                             model.caric_images: caric_images,
                             model.caric_labels: caric_labels}
                sess.run(model.train_op, feed_dict)

                if (step + 1) % 10 == 0:
                    summary, l, acc = sess.run([model.summary_op,
                                                model.loss, model.accuracy],
                                               feed_dict)

                    caric_labels, caric_images = \
                        self.loader.next_group_batch('caric_tr')
                    real_labels, real_images = \
                        self.loader.next_group_batch('real_tr')
                    test_acc, _ = \
                        sess.run(fetches=[model.accuracy, model.loss],
                                 feed_dict={model.real_images: real_images,
                                            model.real_labels: real_labels,
                                            model.caric_images: caric_images,
                                            model.caric_labels: caric_labels})
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]'
                           % (step + 1, self.pretrain_iter, l, acc, test_acc))

                if (step + 1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path,
                                                  'pre_model'),
                               global_step=step + 1)
                    print ('pre_model-%d saved..!' % (step + 1))

    def train(self):
        # load faces
        real_images, real_labels = self.load_real(self.real_dir,
                                                  split='train')
        caric_images, caric_labels = self.load_caric(self.caric_dir,
                                                     split='train')
        combined_images = self.load_combined()
        label_set = set(np.hstack((real_labels, caric_labels)))

        self.loader.add_dataset('real_images', real_images)
        self.loader.add_dataset('caric_images', caric_images)
        self.loader.add_dataset('real_labels', real_labels)
        self.loader.add_dataset('caric_labels', caric_labels)

        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
            # restore variables of F and G
            print ('loading pretrained model F..')
            f_variables_to_restore = \
                slim.get_model_variables(scope='content_extractor')
            f_restorer = tf.train.Saver(f_variables_to_restore)
            f_restorer.restore(sess, self.pretrained_model)

            print ('loading pretrained model G..')
            g_variables_to_restore = \
                slim.get_model_variables(scope='generator')
            g_restorer = tf.train.Saver(g_variables_to_restore)
            g_restorer.restore(sess, self.pretrained_model)

            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print ('start training..!')
            f_interval = 15
            for step in range(self.train_iter + 1):

                i = step % int(real_images.shape[0] / self.batch_size)
                src_images = self.loader.next_batch('real_images')
                pos_ones, pos_twos = self.get_pairs(combined_images,
                                                    label_set,
                                                    set_type='positive')
                neg_ones, neg_twos = self.get_pairs(combined_images,
                                                    label_set,
                                                    set_type='negative')

                feed_dict = {model.src_images: src_images,
                             model.pos_ones: pos_ones,
                             model.pos_twos: pos_twos,
                             model.neg_ones: neg_ones,
                             model.neg_twos: neg_twos}

                for _ in xrange(self.src_disc_rep):
                    sess.run(model.d_train_op_src, feed_dict)
                for _ in xrange(self.src_gen_rep):
                    sess.run([model.g_train_op_src], feed_dict)

                if step > 1600:
                    f_interval = 30

                if i % f_interval == 0:
                    sess.run(model.f_train_op_src, feed_dict)

                if (step + 1) % 10 == 0:
                    summary, dl, gl, fl = sess.run([model.summary_op_src,
                                                    model.d_loss_src,
                                                    model.g_loss_src,
                                                    model.f_loss_src],
                                                   feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Source] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f] f_loss: [%.6f]'
                           % (step + 1, self.train_iter, dl, gl, fl))

                # train the model for target domain T
                # j = step % int(caric_images.shape[0] / self.batch_size)
                # trg_images = caric_images[j * self.batch_size:(j + 1) * self.batch_size]
                trg_images = self.loader.next_batch('caric_images')
                feed_dict = {model.src_images: src_images,
                             model.trg_images: trg_images,
                             model.pos_ones: pos_ones,
                             model.pos_twos: pos_twos,
                             model.neg_ones: neg_ones,
                             model.neg_twos: neg_twos}
                for _ in xrange(self.trg_disc_rep):
                    sess.run(model.d_train_op_trg, feed_dict)
                for _ in xrange(self.trg_gen_rep):
                    sess.run(model.g_train_op_trg, feed_dict)

                if (step + 1) % 10 == 0:
                    summary, dl, gl = sess.run([model.summary_op_trg,
                                                model.d_loss_trg,
                                                model.g_loss_trg],
                                               feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Target] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]'
                           % (step + 1, self.train_iter, dl, gl))

                if (step + 1) % 200 == 0:
                    saver.save(sess, os.path.join(
                        self.model_save_path, 'dtn_ext'), global_step=step + 1)
                    print ('model/dtn_ext-%d saved' % (step + 1))

    def eval(self):
        # build model
        model = self.model
        model.build_model()

        # load real faces
        real_images, _ = self.load_real(self.real_dir)
        self.loader.add_dataset(name='real_images',
                                data_ptr=real_images)

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print ('start sampling..!')
            for i in range(self.sample_iter):
                # train model for source domain S
                batch_images = self.loader.next_batch('real_images')
                feed_dict = {model.images: batch_images}
                sampled_batch_images = sess.run(model.sampled_images,
                                                feed_dict)

                # merge and save source images and sampled target images
                merged = self.merge_images(batch_images, sampled_batch_images)
                path = os.path.join(self.sample_save_path,
                                    'sample-%d-to-%d.png' %
                                    (i * self.batch_size, (i + 1) * self.batch_size))
                scipy.misc.imsave(path, merged)
                print ('saved %s' % path)
