import tensorflow as tf
print("사용 가능한 GPU 개수:", len(tf.config.list_physical_devices('GPU')))