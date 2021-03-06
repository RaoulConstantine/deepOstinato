from deepOstinato.preprocessing.loader import Load_Numpy
from deepOstinato.preprocessing.minmaxnormalizer import MinMaxNormaliser, MinMaxDenormaliser
from deepOstinato.preprocessing.short_time_fourier_transform import ISTFT, Inverse_Mel
from deepOstinato.preprocessing.saver import Saver
import numpy as np

if __name__ == '__main__':

    sample_rate = 22050
    input_path = "raw_data/transformed_audio/"
    output_path = 'raw_data/postproc_wav_files/'

    loaded_audio = Load_Numpy().load_npy_audio(input_path)
    scaler = MinMaxDenormaliser(min_val=-0.5, max_val=0.5)
    scaler.fit()
    loaded_audio = np.array(loaded_audio)
    denormalised_audio = scaler.transform(loaded_audio)

    inversed_audio = Inverse_Mel().imel(denormalised_audio)

    Saver().save_wav(inversed_audio, output_path, sample_rate)
