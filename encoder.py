import process_data
import math
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers
import numpy as np
from functools import partial

LOSS_OUT_FILE = 'Epoch_Loss.txt'

def next_batch(c_batch, batch_size, sess):
    ch1_arr = []
    ch2_arr = []
    wav_arr_ch1, wav_arr_ch2, sample_rate = process_data.get_next_batch(c_batch, batch_size, sess)

    for sub_arr in wav_arr_ch1:
        batch_size_ch1 = math.floor(len(sub_arr)/inputs)
        sub_arr = sub_arr[:(batch_size_ch1*inputs)]
        ch1_arr.append(np.array(sub_arr).reshape(batch_size_ch1, inputs))

    for sub_arr in  wav_arr_ch2:
        batch_size_ch2 = math.floor(len(sub_arr)/inputs)
        sub_arr = sub_arr[:(batch_size_ch2*inputs)]
        ch2_arr.append(np.array(sub_arr).reshape(batch_size_ch2, inputs))

    return np.array(ch1_arr), np.array(ch2_arr), sample_rate

def create_model():
    l2 = 0.0001

    input_size = 12348
    hidden_1_size = 8400
    hidden_2_size = 3440
    hidden_3_size = 2800

    inputs = tf.keras.Input(shape=(None, input_size))

    autoencoder_dnn = partial(layers.Dense,
                              activation = tf.nn.elu,
                              kernel_initializer = initializers.VarianceScaling(),
                              kernel_regularizer = tf.keras.regularizers.l2(l2))

    X = autoencoder_dnn(hidden_1_size)(inputs)
    X = autoencoder_dnn(hidden_2_size)(X)
    X = autoencoder_dnn(hidden_3_size)(X)
    X = autoencoder_dnn(hidden_2_size)(X)
    X = autoencoder_dnn(hidden_1_size)(X)
    outputs = autoencoder_dnn(input_size, activation=None)(X)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

@tf.function
def loss_fn(X, Y):
    reconstruction_loss = tf.reduce_mean(tf.square(Y - X))
    reg_loss = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    return tf.add_n([reconstruction_loss] + reg_loss)

def main():
    process_data.process_wav()

    # Learning rate
    lr = 0.0001

    # Change the epochs variable to define the
    # number of times we iterate through all our batches
    epochs = 1000

    # Change the batch_size variable to define how many songs to load per batch
    batch_size = 50

    # Change the batches variable to change the number of batches you want per epoch
    batches = 1

    model = create_model()

    # reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    # reg_loss = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    # loss = tf.add_n([reconstruction_loss] + reg_loss)

    optimizer = tf.keras.optimizers.Adam(lr)

    model.summary()
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        # metrics=['acc']
    )

    n_batch = 50
    input_size = 12348

    dataset = tf.data.TFRecordDataset(filenames = ['./audio.tfrecords'])

    raw_example = next(iter(dataset))
    parsed = tf.train.Example.FromString(raw_example.numpy())
    feature = parsed.features.feature
    audio_raw = feature['audio_raw'].bytes_list.value[0]
    audio = tf.audio.decode_wav(audio_raw)
    sample_rate = audio.sample_rate
    audio = audio.audio.numpy()

    data = np.hstack([audio[:,0], audio[:,1]])
    data = data[:(input_size*(data.shape[0]//input_size))].reshape((1, (data.shape[0]//input_size), input_size))
    ds = tf.data.Dataset.from_tensor_slices((data, data))
    # ds = tf.data.Dataset.from_tensor_slices((data, data))

    model.fit(ds.batch(1), epochs=3)

    # model.fit(data, labels, epochs=10, batch_size=32)

    return

    # training_op = optimizer.minimize(loss, var_list=model.trainable_variables)
    # init = tf.global_variables_initializer()

    ##### Run training
    with tf.Session() as sess:
        init.run()

        for epoch in range(epochs):
            epoch_loss = []
            print("Epoch: " + str(epoch))
            for i in range(batches):
                ch1_song, ch2_song, sample_rate = next_batch(i, batch_size, sess)
                total_songs = np.hstack([ch1_song, ch2_song])
                batch_loss = []

                for j in range(len(total_songs)):
                    x_batch = total_songs[j]
                    _, l = sess.run([training_op, loss], feed_dict={X:x_batch})
                    batch_loss.append(l)
                    print("Song loss: " + str(l))

                print("Curr Epoch: " + str(epoch) + " Curr Batch: " + str(i) + "/"+ str(batches))
                print("Batch Loss: " + str(np.mean(batch_loss)))
                epoch_loss.append(np.mean(batch_loss))

            print("Epoch Avg Loss: " + str(np.mean(epoch_loss)))

            if epoch % 1000 == 0:
                ch1_song_new, ch2_song_new, sample_rate_new = next_batch(2, 1, sess)
                x_batch = np.hstack([ch1_song_new, ch2_song_new])[0]
                print("Sample rate: " + str(sample_rate_new))

                orig_song = []
                full_song = []
                evaluation = outputs.eval(feed_dict={X: x_batch})
                print("Output: " + str(evaluation))
                full_song.append(evaluation)
                orig_song.append(x_batch)

                # Merge the nested arrays
                orig_song = np.hstack(orig_song)
                full_song = np.hstack(full_song)

                # Compute and split the channels
                orig_song_ch1 = orig_song[:math.floor(len(orig_song)/2)]
                orig_song_ch2 = orig_song[math.floor(len(orig_song)/2):]
                full_song_ch1 = full_song[:math.floor(len(full_song)/2)]
                full_song_ch2 = full_song[math.floor(len(full_song)/2):]

                # Save both the untouched song and reconstructed song to the 'output' folder
                process_data.save_to_wav(full_song_ch1, full_song_ch2, sample_rate, orig_song_ch1, orig_song_ch2, epoch, 'output', sess)

if __name__ == 'main':
    main()
