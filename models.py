from tensorflow.keras.layers import Conv2D,Dense,Flatten,Input, AveragePooling2D, Conv2D,Concatenate
import tensorflow as tf
class Models:
    def ResNet18Tf():
      resnet_input =  Input((32,32,3))
      conv_1 = Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding="same")(resnet_input)


      conv_b1_1 = Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding="same")(conv_1)
      conv_b1_2 = Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding="same")(conv_b1_1)
      sum_0_2 = Concatenate()([conv_1,conv_b1_2])
      conv_b1_3 = Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding="same")(sum_0_2)
      conv_b1_4 = Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding="same")(conv_b1_3)

      sum_1 = Concatenate()([sum_0_2,conv_b1_4])
      avg_1 = AveragePooling2D(pool_size=(2,2))(sum_1)

      conv_b2_1 = Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(avg_1)
      conv_b2_2 = Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(conv_b2_1)
      sum_1_2 = Concatenate()([avg_1,conv_b2_2])
      conv_b2_3 = Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(sum_1_2)
      conv_b2_4 = Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="same")(conv_b2_3)

      sum_2 = Concatenate()([sum_1_2,conv_b2_4])
      avg_2 = AveragePooling2D(pool_size=(2,2))(sum_2)

      conv_b3_1 = Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding="same")(avg_2)
      conv_b3_2 = Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding="same")(conv_b3_1)
      sum_2_2 = Concatenate()([avg_2,conv_b3_2])
      conv_b3_3 = Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding="same")(sum_2_2)
      conv_b3_4 = Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding="same")(conv_b3_3)

      sum_3 = Concatenate()([sum_2_2,conv_b3_4])
      avg_3 = AveragePooling2D(pool_size=(2,2))(sum_3)

      conv_b4_1 = Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding="same")(avg_3)
      conv_b4_2 = Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding="same")(conv_b4_1)
      sum_3_2 = Concatenate()([avg_3,conv_b4_2])
      conv_b4_3 = Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding="same")(avg_3)
      conv_b4_4 = Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding="same")(conv_b4_3)

      sum_4 = Concatenate()([sum_3_2,conv_b4_4])
      avg = AveragePooling2D(pool_size=(2,2))(sum_4)

      flat = Flatten()(avg)#problema <--

      dense1 = Dense(16,activation='relu')(flat)#avg
      dense2 = Dense(10,activation='softmax')(flat)#maxp

      flat = Flatten()(dense2)

      resnet_fix = tf.keras.models.Model(inputs=resnet_input,outputs=dense2)
      return resnet_fix
