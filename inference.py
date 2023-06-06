import sys
import argparse
import time
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["aqua", "gsm8k", "gsmic", "addsub", "multiarith",
                                 "svamp", "singleeq", "math_prealgebra",
                                 "math_algebra", "math_counting_and_probability", "math_geometry",
                                 "math_intermediate_algebra",
                                 "math_number_theory", "math_precalculus"],
                        help="dataset used for experiment"
                        )
    parser.add_argument("--method", type=str, default="holmes+",
                        choices=["few_shot_cot", "auto_cot", "ltm_cot", "key_cot", "holmes", "holmes+", "zero_shot_ps+"], help="method"
                        )
    parser.add_argument("--model", type=str, default='gpt3_chat', choices=["gpt3-003", "gpt3_chat", "gpt3-002"])
    parser.add_argument("--random_seed", type=int, default=1, help="set random seed")
    parser.add_argument("--resume_id", type=int, default=0,
                        help="resume from which question id (current line number in the output file)"
                        )
    parser.add_argument("--num_answers", type=int, default=1, help="the number of answers acquired from API")
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="mini-batch size should be 1 because GPT-3 API takes only 1 input for each request"
                        )
    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    parser.add_argument("--temperature", type=float, default=0, help="range is [0:2]. Higher values means more random.")
    parser.add_argument("--max_length_cot", type=int, default=512,
                        help="maximum length of output tokens by model for reasoning extraction"
                        )
    parser.add_argument("--max_length_direct", type=int, default=512,
                        help="maximum length of output tokens by model for answer extraction"
                        )
    parser.add_argument("--limit_dataset_size", type=int, default=0,
                        help="whether to limit test dataset size. if 0, we use all the samples in the dataset"
                        )
    parser.add_argument("--api_time_interval", type=float, default=1,
                        help="sleep between runs to avoid excedding the rate limit of openai api"
                        )
    parser.add_argument("--log_dir", type=str, default="./log/", help="log directory")
    parser.add_argument(
        "--demo_path", type=str, default="demos/holmes2/stage", help="pre-generated demos used for experiment"
    )
    parser.add_argument("--output_dir", type=str, default="record/holmes+/holmes+gsmic.txt", help="output directory")
    parser.add_argument("--resume_correction", type=int, default=0, help="resume from how many data was correct before")

    args = parser.parse_args()
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset.startswith("math"):
        subtask = args.dataset[5:]
        args.dataset_path = "./dataset/MATH/test/{}/{}.json".format(subtask, subtask)
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "gsmic":
        args.dataset_path = "./dataset/GSM-IC/test.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    else:
        raise ValueError("dataset is not properly defined ...")

    return args

def main():
    args = parse_arguments()

    print('*****************************')
    print(args)
    print('*****************************')

    fix_seed(args.random_seed)



    decoder = Decoder()
    print("setup data loader ...")
    dataloader = setup_data_loader(args)

    if args.method == "key_cot":
        demo_path = args.demo_path + "1"
        fewshot_stage1 = create_fewshot(args, demo_path)
        demo_path = args.demo_path + "2"
        fewshot_stage2 = create_fewshot(args, demo_path)
        demo_path = args.demo_path + "3"
        fewshot_stage3 = create_fewshot(args, demo_path)
        demo_path = args.demo_path + "4"
        fewshot_stage4 = create_fewshot(args, demo_path)
        demo_path = args.demo_path + "5"
        fewshot_stage5 = create_fewshot(args, demo_path)
    elif args.method == "ltm_cot":
        demo_path = args.demo_path + "1"
        fewshot_stage1 = create_fewshot(args, demo_path)
        demo_path = args.demo_path + "2"
        fewshot_stage2 = create_fewshot(args, demo_path)
    elif (args.method == "few_shot_cot") or(args.method == "zero_shot_ps+"):
        demo_path = args.demo_path
        fewshot_stage1 = create_fewshot(args, demo_path)
    elif args.method == "holmes":
        demo_path = args.demo_path + "1"
        fewshot_stage1 = create_fewshot(args, demo_path)
        demo_path = args.demo_path + "2"
        fewshot_stage2 = create_fewshot(args, demo_path)
        demo_path = args.demo_path + "3"
        fewshot_stage3 = create_fewshot(args, demo_path)
        demo_path = args.demo_path + "4"
        fewshot_stage4 = create_fewshot(args, demo_path)
    elif args.method == "holmes+":
        demo_path = args.demo_path + "1"
        fewshot_stage1 = create_fewshot(args, demo_path)
        demo_path = args.demo_path + "2"
        fewshot_stage2 = create_fewshot(args, demo_path)
        demo_path = args.demo_path + "3"
        fewshot_stage3 = create_fewshot(args, demo_path)
    else:
        pass

    if args.resume_id ==0:
        total = 0
    else:
        total = args.resume_id - 1
    correct_list = []
    correct_list.append(args.resume_correction)
    max_length = args.max_length_cot

    with open(args.output_dir, "a") as wp:
        for i, data in enumerate(dataloader):
            if i < args.resume_id - 1:
            # if i < 297:
                continue
            output_line = {}

            print('*************************')
            print("{}st data".format(i + 1))
            wp.write("{}st data".format(i + 1))

            x_, y_ = data
            #x = "Question: " + x_[0] + "\n"
            x = "Q: " + x_[0] + "\n"
            y = y_[0].strip()

            output_line["question"] = x
            output_line["gold_ans"] = y
            key_error_flag = False
            ltm_error_flag = False
            if args.method == "key_cot":

                q_stage12 = x + "A:"
                while True:
                    try:
                        questions = decoder.key_cot_decode(fewshot_stage1, q_stage12, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(5)
                        continue

                output_line["q"] = questions
                while True:
                    try:
                        keys = decoder.key_cot_decode(fewshot_stage2, q_stage12, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(5)
                        continue

                output_line["k"] = keys

                clist = extract_keys(keys)
                #print(clist)
                qlist = extract_questions(questions)
                key_and_q = generate_kq(clist, qlist)
                q_stage3 = x + "Hint: " + key_and_q + "\nA:"
                while True:
                    try:
                        key_location = decoder.key_cot_decode(fewshot_stage3, q_stage3, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(5)
                        continue
                print(key_location)
                key_location, answer_location = recorrect_location(key_location, len(qlist))
                location_dict_key = locate_key(key_location, len(qlist), len(clist))
                location_dict_answer = locate_answer(answer_location, len(qlist))
                q_stage5 = trans2math(clist, x_[0])

                while True:
                    try:
                        trans2math_keys = decoder.key_cot_decode(fewshot_stage5, q_stage5, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(5)
                        continue

                clist_new = extract_keys(trans2math_keys)
                #print(clist_new)

                if len(clist_new) == len(clist):
                    try:
                        hint_stage4 = generate_hint(clist_new, qlist, location_dict_key, location_dict_answer)
                    except KeyError as e:
                        print("Key Error:", e)
                        key_error_flag = True
                else:
                    try:
                        hint_stage4 = generate_hint(clist, qlist, location_dict_key, location_dict_answer)
                    except KeyError as e:
                        print("Key Error:", e)
                        key_error_flag = True

                q_stage4 = x_[0] + "\nQ: " + hint_stage4

                if key_error_flag:
                    answer = 'Occurred key error'
                else:
                    while True:
                        try:
                            answer = decoder.key_cot_decode(fewshot_stage4, q_stage4, args, max_length)
                            break
                        except Exception as e:
                            print("api Error:", e)
                            time.sleep(10)
                            continue

                print(q_stage4)
                print(answer)
                print("\n")
            elif args.method == "ltm_cot":
                q_stage1 = x + "A:"
                while True:
                    try:
                        sub_q = decoder.key_cot_decode(fewshot_stage1, q_stage1, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(3)
                        continue

                output_line["sub_questions"] = sub_q
                print(sub_q)

                subq_list = extract_question(sub_q)
                if len(subq_list)!=0:
                    print(subq_list)

                    q_stage2 = x + "A:Let's break down this question:"
                    j = 1
                    for q in subq_list:
                        q_stage2 += ' ' + str(j) + '.' + q
                        j += 1
                    q_stage2 += '\n'
                    while True:
                        try:
                            answer = decoder.key_cot_decode(fewshot_stage2, q_stage2, args, max_length)
                            break
                        except Exception as e:
                            print("api Error:", e)
                            time.sleep(3)
                            continue
                    print(answer)
                    print("\n")
                else:
                    print("generate sub question wrong!")
                    ltm_error_flag = True
            elif args.method == "holmes":

                q_stage1 = 'Q: Rewrite this problem by removing information which is unnecessary for solving the final question: \"'+x_[0]+'\"\nA:'
                while True:
                    try:
                        new_q = decoder.key_cot_decode(fewshot_stage1, q_stage1, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:",e)
                        time.sleep(2)
                        continue

                print(q_stage1)

                q_stage2 = "Q:" + new_q + "\nA:"
                while True:
                    try:
                        answer = decoder.key_cot_decode(fewshot_stage2, q_stage2, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(2)
                        continue

                print(q_stage2)
                print(answer)
                print("\n")
            elif args.method == "holmes+":
                q_stage1 = x + "A:"
                while True:
                    try:
                        keys = decoder.key_cot_decode(fewshot_stage1, q_stage1, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(1)
                        continue
                print(keys)
                output_line["k"] = keys
                clist = extract_keys(keys)
                q_stage2 = trans2math(clist, x_[0])

                while True:
                    try:
                        trans2math_keys = decoder.key_cot_decode(fewshot_stage2, q_stage2, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(1)
                        continue

                clist_new = extract_keys(trans2math_keys)
                hint = "With the Equation Hints:"
                for c in clist_new:
                    hint += " '" + c + "',"
                hint += " we will answer the question."
                q_stage3 = x + "A: " + hint
                while True:
                    try:
                        answer = decoder.key_cot_decode(fewshot_stage3, q_stage3, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(1)
                        continue
                print(q_stage3)
                print(answer)
                print("\n")
            elif args.method == "few_shot_cot":
                q = x + "A:"
                while True:
                    try:
                        answer = decoder.key_cot_decode(fewshot_stage1, q, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(1)
                        continue
                print(q)
                print(answer)
                print("\n")
            elif args.method == "zero_shot_ps+":
                q = x + "A: " + fewshot_stage1
                while True:
                    try:
                        answer0 = decoder.key_cot_decode("", q, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(1)
                        continue
                z = "\n".join([q, answer0, args.direct_answer_trigger])
                while True:
                    try:
                        answer = decoder.key_cot_decode("", z, args, max_length)
                        break
                    except Exception as e:
                        print("api Error:", e)
                        time.sleep(1)
                        continue

            if key_error_flag:
                pred = 'Wrong answer because of wrong format answer of LLM'
            elif ltm_error_flag:
                pred = 'Wrong sub question list because of wrong format answer of LLM'
            else:
                pred = answer_cleaning(args, answer)

            output_line["pred_ans"] = pred



            # Choose the most frequent answer from the list ...
            print("pred : {}".format(pred))
            print("GT : " + y)
            print('*************************')

            # Checking answer ...
            # correct = (np.array([pred]) == np.array([y])).sum().item()
            if not args.dataset.startswith("math"):
                y = y.replace(",", "")

            if args.dataset.startswith("math"):
                correct = int(is_equiv(pred, y))
            elif pred == '':
                correct = 0
            elif key_error_flag or ltm_error_flag:
                correct = 0
            elif float(pred)==float(y):
                correct = 1
            else:
                correct = 0
            correct_list.append(correct)
            total += 1  # np.array([y]).size(0)

            accuracy = (sum(correct_list) * 1.0 / total) * 100
            print("correct number : {}".format(sum(correct_list) * 1.0))
            print("accuracy : {}".format(accuracy))
            wp.write(": correct? {}".format(correct))
            wp.write(": correct_number? {}".format(sum(correct_list) * 1.0))
            wp.write(": accuracy? {}\n".format(accuracy))

            if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
                break
                # raise ValueError("Stop !!")

            # Calculate accuracy ...
        #accuracy = (sum(correct_list) * 1.0 / total) * 100
        #print("accuracy : {}".format(accuracy))

if __name__ == "__main__":
    main()

