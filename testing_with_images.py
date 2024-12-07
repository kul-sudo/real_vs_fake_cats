from os import listdir
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from config import *
import matplotlib.pyplot as plt
import matplotlib
from keras.applications.resnet50 import preprocess_input

matplotlib.rcParams["toolbar"] = "None"  # Remove the buttons & other UI

FONT_SIZE = 28
TEST_PATH = "Demo/"

n_accurate_predictions = 0
n_predictions = 0


def show_image(cat_path, real, result):
    global n_predictions, n_accurate_predictions

    # Print the information on the accuracy of predictions
    prediction_is_real = result >= 0.5
    prediction_is_correct = prediction_is_real == real

    n_predictions += 1
    if prediction_is_correct:
        n_accurate_predictions += 1

    print(
        f"{round((n_accurate_predictions/n_predictions)*100)}% of correct predictions."
    )

    plt.figure(num=" ")
    plt.imshow(plt.imread(cat_path))

    who = f'{"real" if prediction_is_real else "fake"} cat'
    plt.gca().set_title(who, fontsize=FONT_SIZE)

    percent = round(result * 100)
    reliability = percent if result >= 0.5 else 100 - percent
    reliability_text = (
        f"{reliability}% sure ({"correct" if prediction_is_correct else "incorrect"})"
    )

    plt.figtext(0.5, 0.05, reliability_text, ha="center", fontsize=FONT_SIZE)

    plt.axis("off")
    plt.show()
    plt.close()


def test(cat_path, real):
    test_image = image.load_img(cat_path, target_size=IMAGE_SIZE)

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    test_image = preprocess_input(test_image)

    result = model.predict(test_image, verbose=0)[0][0]
    show_image(cat_path, real, result)


model = load_model("epoch=22 val_accuracy=0.981 val_loss=0.114.keras")

for folder in ["real-cats", "ai-cats"]:
    fnames_cats = listdir(f"{TEST_PATH}/{folder}")

    for fnames in fnames_cats:
        test(f"{TEST_PATH}{folder}/{fnames}", folder == "real-cats")
