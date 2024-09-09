import re
from nltk import ngrams
import numpy as np

from config import alphabet


def extract_phrases(text):
    """
    Sử dụng biểu thức chính quy để tìm các cụm từ\n
    Ex: "Hello world! This is a test." -> ["Hello world", "This is a test"]
    """
    text = ''.join([c for c in text if c in alphabet])
    return re.findall(r'\w[\w ]*|\s\W+|\W+', text)

def gen_ngrams(words, n=5):
    """
    Sử dụng hàm ngrams để tạo ra các n-gram từ chuỗi words.\n
    N-gram là một chuỗi gồm n phần tử liên tiếp trong một chuỗi lớn hơn.\n
    Ví dụ, nếu words là "I am learning Python", và n là 3, thì các n-gram sẽ là "I am learning", "am learning Python".
    """
    return ngrams(words.split(), n)

def batch_to_device(text, tgt_input, tgt_output, tgt_padding_mask, device):
    """
    Chuyển đổi các đối tượng dữ liệu sang thiết bị được chỉ định\n
    param: device - thiết bị đích (cpu/gpu)
    """
    
    # non_blocking=True - cho phép việc chuyển đổi diễn ra một cách không chặn đầu vào/đầu ra của hàm
    text = text.to(device, non_blocking=True)
    tgt_input = tgt_input.to(device, non_blocking=True)
    tgt_output = tgt_output.to(device, non_blocking=True)
    tgt_padding_mask = tgt_padding_mask.to(device, non_blocking=True)

    return text, tgt_input, tgt_output, tgt_padding_mask

def get_bucket(w):
    bucket_size = (w // 5) * 5

    return bucket_size

def compute_accuracy(ground_truth, predictions, mode='full_sequence'):
    """
    Hàm tính toán độ chính xác giữa dữ liệu thực và dự đoán
    """
    if mode == 'per_char':

        accuracy = []

        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            total_count = len(label)
            correct_count = 0
            try:
                for i, tmp in enumerate(label):
                    if tmp == prediction[i]:
                        correct_count += 1
            except IndexError:
                continue
            finally:
                try:
                    accuracy.append(correct_count / total_count)
                except ZeroDivisionError:
                    if len(prediction) == 0:
                        accuracy.append(1)
                    else:
                        accuracy.append(0)
        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    elif mode == 'full_sequence':
        try:
            correct_count = 0
            for index, label in enumerate(ground_truth):
                prediction = predictions[index]
                if np.all(prediction == label):
                    correct_count += 1
            avg_accuracy = correct_count / len(ground_truth)
        except ZeroDivisionError:
            if not predictions:
                avg_accuracy = 1
            else:
                avg_accuracy = 0

    elif mode == 'word':
        accuracy = []

        for index, label in enumerate(ground_truth):
            prediction = predictions[index]

            gt_word_list = label.split(' ')
            pred_word_list = prediction.split(' ')

            for i, gt_w in enumerate(gt_word_list):
                if i > len(pred_word_list) - 1:
                    accuracy.append(0)
                    continue

                pred_w = pred_word_list[i]

                if pred_w == gt_w:
                    accuracy.append(1)
                else:
                    accuracy.append(0)

        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    else:
        raise NotImplementedError('Other accuracy compute mode has not been implemented')

    return avg_accuracy