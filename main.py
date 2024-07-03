import streamlit as st
import io
import skimage
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sklearn
warnings.filterwarnings('ignore')
sklearn.set_config(transform_output='pandas')


st.title('Сожми картинку с SVD!')

image = st.file_uploader(
    label='Загрузи картинку для SVD-сжатия',
    type=['png', 'jpg'])


if image:
    image = skimage.io.imread(image)
    image = image[:, :, 1]
    st.image(image=image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    U, sing_nums, V = np.linalg.svd(image)
    singulars_n = sing_nums.size
    sigma = np.zeros(shape=image.shape)
    np.fill_diagonal(sigma, sing_nums)

    k = st.number_input(label=f'Введите количество сингулярных чисел от 1 до {singulars_n}',
                        min_value=1,
                        max_value=singulars_n)

    trunc_U = U[:, :k]
    trunc_sigma = sigma[:k, :k]
    trunc_V = V[:k, :]
    svd_image = trunc_U @ trunc_sigma @ trunc_V
    st.write(svd_image)
    st.write(f'k = {k}')

    svd_image = np.squeeze(svd_image)

    # Используем matplotlib для отображения изображения
    fig, ax = plt.subplots()
    ax.imshow(svd_image, cmap='gray')
    ax.axis('off')  # Отключаем оси для более чистого отображения

    # Выводим изображение с помощью streamlit
    st.pyplot(fig)

    #  Сохраненине изображения для скачивания
    def saving_img():
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf

    download_image = st.download_button(
        label='Скачай результат!',
        data=saving_img(),
        file_name='svd_image.png',
        mime='image/png')





