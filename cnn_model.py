import tensorflow as tf
import numpy as np

class cnn_classify_model():
    def __init__(self,train_iterator,infer_iterator):
        self.train_iterator=train_iterator
        self.infer_iterator=infer_iterator
        self.build_graph()
    def build_graph(self):
        self.input_x=tf.placeholder(shape=[None,None],dtype=tf.int32)
        self.input_y=tf.placeholder(shape=[None,2],dtype=tf.int32)
        with tf.device('/gpu:0'):
            embedding_layer=tf.get_variable(initializer=tf.truncated_normal([160277,512],stddev=0.1),name="embedding_layer")
            embedding_input=tf.nn.embedding_lookup(embedding_layer,self.input_x)
        layers_outputs=[]
        for n_gram in [4,8,16]:
            with tf.name_scope('cnn_%d'%n_gram):
                convolution_outputs=tf.layers.conv1d(embedding_input,256,n_gram,activation=tf.nn.relu)
                max_pooling_outputs=tf.reduce_max(convolution_outputs,axis=1)
                layers_outputs.append(max_pooling_outputs)
        cnn_outputs=tf.concat(layers_outputs,1)
        with tf.name_scope("dropout"):
            cnn_dropouts=tf.nn.dropout(cnn_outputs,0.8)
        with tf.name_scope('fcnn'):
            fcnn_weight=tf.get_variable(initializer=tf.truncated_normal([768,2],stddev=0.1),name="fcnn_weight")
            fcnn_bias=tf.get_variable(initializer=tf.zeros([2]),name="fcnn_bias")
            logits=tf.matmul(cnn_dropouts,fcnn_weight)+fcnn_bias
            self.pred=tf.nn.softmax(logits)
        with tf.name_scope('loss'):
            nn_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.input_y))
            weight_loss=tf.nn.l2_loss(fcnn_weight)+tf.nn.l2_loss(fcnn_bias)
            self.loss=nn_loss+weight_loss
            optimizer=tf.train.AdamOptimizer()
            self.loss_op=optimizer.minimize(self.loss)
        with tf.name_scope('acc'):
            correct_pred=tf.equal(tf.argmax(self.pred,1),tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        self.saver=tf.train.Saver(tf.global_variables())
    def train(self,sess,step):
        input_x,input_y=sess.run((self.train_iterator.input_x,self.train_iterator.input_y))
        train_dict={self.input_x:input_x,self.input_y:input_y}
        res=sess.run((self.loss,self.loss_op,self.accuracy),feed_dict=train_dict)
        print("loss:",res[0],"accuracy:",res[2],"step:",step)
    def val_infer(self,sess):
        input_x,input_y=sess.run((self.infer_iterator.input_x,self.infer_iterator.input_y))
        train_dict={self.input_x:input_x}
        return sess.run(self.pred,feed_dict=train_dict),input_y
    def test_infer(self,sess):
        input_x=sess.run(self.infer_iterator.input_x)
        train_dict={self.input_x:input_x}
        return sess.run(self.pred,feed_dict=train_dict)
    def save_model(self,sess,model_name):
        self.saver.save(sess,model_name)
    def load_model(self,sess,model_name):
        self.saver.restore(sess,model_name)
    def debug(self,sess):
        input_x,input_y=sess.run((self.train_iterator.input_x,self.train_iterator.input_y))
        train_dict={self.input_x:input_x,self.input_y:input_y}
        res=sess.run((self.result),feed_dict=train_dict)
        return res