from statistics import mean
from torch.utils.data import Dataset
from nltk import sent_tokenize as sent_tk
import openai
import sys
import json
import torch
import glob
import numpy as np
import pandas as pd
import random
import re
import multiprocessing
import time



'''
openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)
'''


def decoder_for_gpt3(fewshot, question, args, max_length):

    time.sleep(args.api_time_interval)

    # https://beta.openai.com/account/api-keys

    if args.model == 'gpt3-003':
        engine = "text-davinci-003"
    elif args.model == 'gpt3-002':
        engine = "text-davinci-002"
    elif args.model == 'gpt3_chat':
        engine = "gpt-3.5-turbo"
    else:
        raise ValueError("model is not properly defined ...")

    if engine == 'text-davinci-003' or engine == 'text-davinci-002':
        response = openai.Completion.create(
            model=engine,
            prompt=fewshot + question,
            max_tokens=max_length,
            temperature=args.temperature,
            top_p=1,
            n=args.num_answers,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        if args.num_answers != 1:
            answers = []
            for i in range(args.num_answers):
                answers.append(response["choices"][i]["text"])
            answer = self_consistency(answers)
        else:
            answer = response["choices"][0]["text"]
    # gpt-3.5-turbo and gpt-4
    else:
        if args.dataset.startswith("math"):
            format_prompt = """Answer the final question Q according to the given prompts.
                      You should append a sentence like \"The answer is [Your Answer].\" at the end of your output.\n"""
        else:
            format_prompt = ""
        content = format_prompt + fewshot + question
        response = openai.ChatCompletion.create(
            model=engine,
            messages=[
                #"role": "system", "content": "Follow the given examples and answer the question."},
                {"role": "user", "content": content}
            ],
            max_tokens=max_length,
            temperature=args.temperature,
            top_p=1,
            n=args.num_answers,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        if args.num_answers != 1:
            answers = []
            for i in range(args.num_answers):
                answers.append(response["choices"][i]["message"]["content"])
            answer = self_consistency(answers)
        else:
            answer = response["choices"][0]["message"]["content"]

    return answer


# self-consistency-for-key&question
def self_consistency(candidate):
    N = len(candidate)
    return candidate[0]


class Decoder:
    def __init__(self):
        # print_now()
        pass

    def key_cot_decode(self, fewshot, input, args, max_length):
        response = decoder_for_gpt3(fewshot, input, args, max_length)
        return response

def create_fewshot(args, demo_path):
    x, z, y = [], [], []
    if args.method == "key_cot":
        with open(demo_path, encoding="utf-8") as f:
            json_data = json.load(f)
            json_data = json_data["demo"]
            demo_text = ""
            if ("stage1" in demo_path) or ("stage2" in demo_path):
                for line in json_data:
                    x.append(line["question"])
                    y.append(line["answer"])
                index_list = list(range(len(x)))
                for i in index_list:
                    demo_text += x[i] + " " + y[i] + "\n\n"
            elif "stage3" in demo_path:
                for line in json_data:
                    x.append(line["question"])
                    z.append(line["hint"])
                    y.append(line["answer"])
                index_list = list(range(len(x)))
                for i in index_list:
                    demo_text += x[i] + " " + z[i] + " " + y[i] + "\n\n"
            else:
                for line in json_data:
                    x.append(line["question"])
                    z.append(line["division"])
                    y.append(line["answer"])
                index_list = list(range(len(x)))
                for i in index_list:
                    demo_text += x[i] + " " + z[i] + " " + y[i] + "\n\n"
    elif (args.method == "ltm_cot") or (args.method == "few_shot_cot"):
        with open(demo_path, encoding="utf-8") as f:
            json_data = json.load(f)
            json_data = json_data["demo"]
            demo_text = ""
            for line in json_data:
                x.append(line["question"])
                y.append(line["answer"])
            index_list = list(range(len(x)))
            for i in index_list:
                demo_text += x[i] + " " + y[i] + "\n\n"
    elif args.method == "holmes":
        with open(demo_path, encoding="utf-8") as f:
            json_data = json.load(f)
            json_data = json_data["demo"]
            demo_text = ""
            for line in json_data:
                x.append(line["question"])
                y.append(line["answer"])
                if ("stage4" in demo_path):
                    z.append(line["division"])
                # elif ("stage2" in demo_path):
                #     z.append(line["hint"])
            index_list = list(range(len(x)))
            if ("stage1" in demo_path):
                for i in index_list:
                    demo_text += 'Q: Rewrite this problem by removing information which is unnecessary for solving the final question: \"'+x[i]+'\"\nA: ' + y[i] + "\n\n"
            elif ("stage4" in demo_path):
                for i in index_list:
                    demo_text += x[i] + " " + z[i] + " " + y[i] + "\n\n"
            else:
                for i in index_list:
                    demo_text += x[i] + " " + y[i] + "\n\n"
    elif args.method == "holmes+":
        with open(demo_path, encoding="utf-8") as f:
            json_data = json.load(f)
            json_data = json_data["demo"]
            demo_text = ""
            for line in json_data:
                x.append(line["question"])
                y.append(line["answer"])
                if "stage2" in demo_path:
                    z.append(line["division"])
            index_list = list(range(len(x)))
            if "stage2" in demo_path:
                for i in index_list:
                    demo_text += x[i] + " " + z[i] + " " + y[i] + "\n\n"
            else:
                for i in index_list:
                    demo_text += x[i] + " " + y[i] + "\n\n"
    elif args.method == "zero_shot_ps+":
        with open(demo_path, encoding="utf-8") as f:
            demo_text = f.read()
    else:
        demo_text = ""
    return demo_text


def shuffleDict(d):
  keys = list(d.keys())
  random.shuffle(keys)
  #[(key, d[key]) for key in keys]
  random.shuffle(keys)
  #[(key, d[key]) for key in keys]
  random.shuffle(keys)
  keys = [(key, d[key]) for key in keys]
  #keys = d(keys)
  return dict(keys)


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def data_reader(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + " " + choice)
                answers.append(json_res["correct"])

    elif args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1])

    elif args.dataset in ("addsub", "multiarith", "singleeq", "multiarithic", "singleeqic"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset.startswith("math"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["problem"]
                a = remove_boxed(last_boxed_only_string(line["solution"]))
                questions.append(q)
                answers.append(a)

    elif args.dataset == "gsmic":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["new_question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)



    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers


class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output

def setup_data_loader(args):
    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2 ** 32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = MyDataset(args)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=True,
                                             batch_size=args.minibatch_size,
                                             drop_last=False,
                                             num_workers=dataloader_num_workers,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             pin_memory=True)

    return dataloader


def answer_cleaning(args, pred, must_choice=False):
    preds = pred.split("\n")
    pred = preds[-1]

    if args.method in ("few_shot", "few_shot_cot", "auto_cot", "key_cot", "ltm_cot", "holmes", "holmes+"):
        preds = re.split(r"[Tt]he answer is ", pred)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

    if args.dataset =="aqua":
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq", "gsmic", "multiarithic", "singleeqic"):
        if must_choice:
            pred = re.findall(r'A|B|C|D', pred)
        else:
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset.startswith("math"):
        print(pred)
        pat = re.compile(r"\\boxed{.*}")
        boxed_span = pat.search(pred, 1)
        if answer_flag:
            if pred[-1] == ".":
                pred = pred[:-1]
            pred = re.sub("[$]", "", pred)
            span = re.search(r"\\boxed{.*}", pred)
            if span:
                span = span.span()
                pred = pred[span[0] + 7:span[1] - 1]
            pred = [pred]
        elif boxed_span:
            span = boxed_span.span()
            pred = [pred[span[0] + 7:span[1] - 1]]
        else:
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot_cot", "auto_cot", "key_cot", "ltm_cot", "holmes", "holmes+"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method =="zero_shot_ps+":
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    print("pred_after : " + pred)

    return pred


def recorrect_location(key_location, qnum):
    location_list = key_location.split(";")
    new_key_location = ""
    new_answer_location = ""
    if location_list[-1] == '':
        location_list = location_list[:-1]
    for l in location_list:
        half_list = l.split("the key information for")
        if len(half_list)!= 2:
            continue
        if ('final question' in half_list[1]) or ('final answer' in half_list[1]):
            half_list[1] += str(qnum)
        answer_list = []
        condi_list = []
        if "condition" in half_list[0]:
            if "answer" in half_list[0]:
                if half_list[0].find('condition') < half_list[0].find('answer'):
                    key_condition_and_answer = half_list[0].split('answer')
                    condi_list = re.findall('\d+', key_condition_and_answer[0])
                    answer_list = re.findall('\d+', key_condition_and_answer[1])
                else:
                    key_condition_and_answer = half_list[0].split('condition')
                    condi_list = re.findall('\d+', key_condition_and_answer[1])
                    answer_list = re.findall('\d+', key_condition_and_answer[0])
            else:
                condi_list = re.findall('\d+', half_list[0])
        else:
            if "answer" in half_list[0]:
                answer_list = re.findall('\d+', half_list[0])
            else:
                continue

        if len(condi_list) > 0:
            for c in condi_list:
                new_sentence = "condition " + c + " is the key information for" + half_list[1] + "; "
                new_key_location += new_sentence

        if len(answer_list) > 0:
            for a in answer_list:
                new_sentence = "the answer of " + a + " is the key information for" + half_list[1] + "; "
                new_answer_location += new_sentence

    return new_key_location, new_answer_location


def locate_key(key_location, q_num, c_num):
    location_list = key_location.split(";")
    location_dict = {}
    for i in range(q_num):
        location_dict[i+1]=[]
    if location_list[-1] == '':
        location_list = location_list[:-1]
    for location in location_list:
        sigle_key_with_questions = re.findall('\d+',location)
        sigle_key_with_questions = [int(i) for i in sigle_key_with_questions]
        for subq in sigle_key_with_questions[1:]:
            if (subq <= q_num) & (sigle_key_with_questions[0] <= c_num):
                location_dict[subq].append(sigle_key_with_questions[0])
    return location_dict


def locate_answer(answer_location, q_num):
    location_list = answer_location.split(";")
    location_dict = {}
    for i in range(q_num):
        location_dict[i+1] = []
    if location_list[-1] == '':
        location_list = location_list[:-1]
    for location in location_list:
        sigle_answer_with_questions = re.findall('\d+',location)
        sigle_answer_with_questions = [int(i) for i in sigle_answer_with_questions]
        for subq in sigle_answer_with_questions[1:]:
            if (subq <= q_num) & (sigle_answer_with_questions[0] <= q_num):
                location_dict[subq].append(sigle_answer_with_questions[0])
    return location_dict


def extract_questions(questions):
    sentence1 = re.findall('"(.*?)"', questions)
    question_list = []
    for question in sentence1:
        question_list.append(question)
    #question_list.append(sentence1[0])
    return question_list


def extract_keys(keys):
    sentence1 = re.findall('"(.*?)"', keys)
    condition_list = []
    for condition in sentence1:
        condition_list.append(condition)
    return condition_list


def generate_kq(condition_list, question_list):
    key_and_q = "To answer this question, we need to notice these conditions:"
    for i in range(len(condition_list)):
        if i==len(condition_list)-1:
            key_and_q += " " + str(i+1) + ".'" + condition_list[i] + "'; Having these conditions, then we need to solve these questions:"
        else:
            key_and_q += " " + str(i+1) + ".'" + condition_list[i] + "',"
    for i in range(len(question_list)):
        if i==len(question_list)-1:
            key_and_q += " " + str(i+1) + ".'" + question_list[i] + "';"
        else:
            key_and_q += " " + str(i+1) + ".'" + question_list[i] + "',"

    return key_and_q


def generate_hint(condition_list, question_list, location_dict_key, location_dict_answer):
    hint = "Only rely on hints to solve problems one by one, ended with 'the answer is ...':"
    for i in range(len(question_list)):
        hint += " " + str(i+1) + "." + question_list[i]
        if (len(location_dict_key[i+1]) == 0) & (len(location_dict_answer[i+1]) == 0):
            pass
        elif (len(location_dict_key[i+1]) != 0) & (len(location_dict_answer[i+1]) == 0):
            hint += "(Hint: Notice that "
            for j in range(len(location_dict_key[i+1])):
                hint += "'" + condition_list[location_dict_key[i+1][j]-1] + "'"
                if j == len(location_dict_key[i+1])-1:
                    hint += ")"
                else:
                    hint += ", "
        elif (len(location_dict_key[i+1]) == 0) & (len(location_dict_answer[i+1]) != 0):
            hint += "(Hint: Notice the answer of question "
            for j in range(len(location_dict_answer[i+1])):
                hint += str(location_dict_answer[i+1][j])
                if j == len(location_dict_answer[i+1])-1:
                    hint += ")"
                else:
                    hint += ", "
        else:
            hint += "(Hint: Notice that "
            for j in range(len(location_dict_key[i+1])):
                hint += "'" + condition_list[location_dict_key[i+1][j]-1] + "', "
            hint += "and the answer of question"
            for j in range(len(location_dict_answer[i+1])):
                hint += str(location_dict_answer[i+1][j])
                if j == len(location_dict_answer[i+1])-1:
                    hint += ")"
                else:
                    hint += ", "
        if i != len(question_list)-1:
            hint += ","
        else:
            hint += "\nA:"
    return hint



def extract_question(sub_q):
    information = sub_q.split("we need to know:")
    infer1 = information[0]
    infer2 = information[-1]
    questions = re.findall('"(.*?)"', infer2)
    final_question = re.findall('"(.*?)"', infer1)
    subq_list = []
    for q in questions:
        subq_list.append(q)
    if len(final_question)!=0:
        subq_list.append(final_question[0])
    return subq_list


def trans2math(key_list, question):
    target_question = question + '\nQuestion: Translate following sentences into equation:'
    for i in range(len(key_list)):
        target_question += " " + str(i+1) + ".'" + key_list[i] + "'"

    target_question += "\nA:"

    return target_question


# utils for MATH dataset

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2


# Data sampling for GSM-IC dataset
def sample(df, n_sample=10):
    groups = df.groupby(["original_question"])
    sampled_data = []
    for _, group in groups:
        sampled_data.extend(random.choices(group.to_dict("records"), k=n_sample))
    return sampled_data


def GSMICSampling(rseed=42):
    # Reading data from these paths
    path1 = "./dataset/GSM-IC/GSM-IC_2step.json"
    path2 = "./dataset/GSM-IC/GSM-IC_mstep.json"
    data1 = []
    data2 = []
    with open(path1, "r+", encoding="utf8") as f:
        json_data = json.load(f)
        data1.extend(json_data)
    with open(path2, "r+", encoding="utf8") as f:
        json_data = json.load(f)
        data2.extend(json_data)
    print(len(data1), len(data2))
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df1.head()

    random.seed(rseed)
    # save 2-step test data
    sampled_data_1 = sample(df1)
    len(sampled_data_1)
    with open("./dataset/GSM-IC/test_2step.json", "w", encoding="utf8") as f:
        json.dump(sampled_data_1, f, indent=4)
    # save multi-step test data
    sampled_data_2 = sample(df2)
    len(sampled_data_2)
    with open("./dataset/GSM-IC/test_mstep.json", "w", encoding="utf8") as f:
        json.dump(sampled_data_2, f, indent=4)
    # save all test data
    with open("./dataset/GSM-IC/test.json", "w", encoding="utf8") as f:
        json.dump(sampled_data_1 + sampled_data_2, f, indent=4)

  def make_mic_dataset(input_data, output_data_dir, ic_list, n_ic = 2, n_sample = 100):
    '''
    input_data: name of input dataset, it should be one of ['addsub', 'svamp']
    output_data: path of output dataset
    ic_list: a list of irrelevant contexts if argument is a list; 
    path of json data containing irrelevant contexts if argument is a string
    n_ic: number of irrelevant context, an integar or list
    n_sample: number of items to sample
    '''
    if input_data == "addsub":
        path = "./dataset/AddSub/AddSub.json"
        question_key = "sQuestion"
        offset = 1
    elif input_data == "svamp":
        path = "./dataset/SVAMP/SVAMP.json"
        question_key = "Body"
        offset = 0
    else:
        raise ValueError("dataset is not properly defined ...")
    
    if isinstance(ic_list, str):
        with open(ic_list, "r+", encoding="utf8") as f:
            ic_list = json.load(f)
    elif isinstance(ic_list, list):
        pass
    else:
        raise ValueError("ic_list should be list or string.")
    
    if not isinstance(n_ic, list):
        n_ic = [n_ic]

    with open(path, "r+", encoding="utf8") as f:
        json_data = json.load(f)

    used_data = random.sample(json_data, k = n_sample)
    
    for n in n_ic:
        used_ic = random.sample(ic_list, k = n)
        noisy_data = []
        for item in used_data:
            sents = sent_tk(item[question_key])
            for i in range(n):
                insert_idx = random.randint(0, len(sents) - offset)
                sents.insert(insert_idx, used_ic[i])
            item[question_key] = " ".join(sents)
            noisy_data += [item]
        output_data = output_data_dir + "/{}{}.json".format(input_data, str(n))
        with open(output_data, "w+", encoding="utf8") as f:
            print("Writing noisy data into {}".format(output_data))
            json.dump(noisy_data, f, indent=4)
