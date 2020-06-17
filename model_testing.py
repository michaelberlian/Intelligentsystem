import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import cv2 

#prepare the image
def prepare (filepath):
    IMG_SIZE = 28
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray')
    plt.show()
    return new_array.reshape(-1, 28, 28, 1)

# do the prediction
def predict(path):
    CATEGORIES = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",] 
    model = tf.keras.models.load_model('models/1-conv-2-dense')
    
    # x_test = tf.keras.utils.normalize(x_test, axis=1)
    
    predictions = model.predict([prepare(path)])
    
    for i in range (26):
        print ("{} : {}".format(CATEGORIES[i], predictions[0][i]))
    print("""
    

 
result :""".format(predictions[0][np.argmax(predictions[0])]), CATEGORIES[np.argmax(predictions[0])])
