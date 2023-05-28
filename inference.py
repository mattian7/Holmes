import sys
import argparse
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["aqua", "gsm8k", "gsmic", "commonsensqa", "addsub", "multiarith",
                                 "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters", "math_prealgebra",
                                 "math_algebra", "math_counting_and_probability", "math_geometry", "intermediate_algebra",
                                 "math_number_theory", "math_precalculus"],
                        help="dataset used for experiment"
                        )
    parser.add_argument("--method", type=str, default="key_cot",
                        choices=["zero_shot_cot", "few_shot_cot", "auto_cot", "ltm_cot", "key_cot", "tree_cot"], help="method"
                        )
    parser.add_argument("--model", type=str, default='gpt3_chat', choices=["gpt3", "gpt3_chat"])
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
    parser.add_argument("--max_length_direct", type=int, default=32,
                        help="maximum length of output tokens by model for answer extraction"
                        )
    parser.add_argument("--limit_dataset_size", type=int, default=0,
                        help="whether to limit test dataset size. if 0, we use all the samples in the dataset"
                        )
    parser.add_argument("--api_time_interval", type=float, default=8,
                        help="sleep between runs to avoid excedding the rate limit of openai api"
                        )
    parser.add_argument("--log_dir", type=str, default="./log/", help="log directory")
    parser.add_argument(
        "--demo_path", type=str, default="demos/keycot3/stage", help="pre-generated demos used for experiment"
    )
    parser.add_argument("--output_dir", type=str, default="experiment/gsm8k", help="output directory")

    args = parser.parse_args()
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "gsmic":
        args.dataset_path = "./dataset/GSM-irrelevant-context/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset.startswith("math"):
        subtask = args.dataset[5:]
        args.dataset_path = "./dataset/MATH/test/{}".format(subtask)
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    return args

def main():
    args = parse_arguments()

    print('*****************************')
    print(args)
    print('*****************************')

    fix_seed(args.random_seed)
    #openai.api_key = "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"
    #print("OPENAI_API_KEY:")
    #print(os.getenv("OPENAI_API_KEY")[0:5] + '**********')
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
    elif args.method == "few_shot_cot":
        demo_path = args.demo_path
        fewshot_stage1 = create_fewshot(args, demo_path)
    else:
        pass

    total = 0
    correct_list = []
    max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct

    with open(args.output_dir, "w") as wp:
        for i, data in enumerate(dataloader):
            if i < args.resume_id - 1:
            # if i < 297:
                continue
            output_line = {}

            print('*************************')
            print("{}st data".format(i + 1))

            x_, y_ = data
            x = "Q: " + x_[0] + "\n"
            y = y_[0].strip()

            output_line["question"] = x
            output_line["gold_ans"] = y

            if args.method == "key_cot":

                q_stage12 = x + "A:"
                questions = decoder.key_cot_decode(fewshot_stage1, q_stage12, args, max_length)
                output_line["q"] = questions

                keys = decoder.key_cot_decode(fewshot_stage2, q_stage12, args, max_length)
                output_line["k"] = keys

                clist = extract_keys(keys)
                print(clist)
                qlist = extract_questions(questions)
                key_and_q = generate_kq(clist, qlist)
                q_stage3 = x + "Hint: " + key_and_q + "\nA:"
                key_location = decoder.key_cot_decode(fewshot_stage3, q_stage3, args, max_length)
                key_location = recorrect_location(key_location)
                location_dict = locate_key(key_location, len(qlist) - 1, len(clist))
                q_stage5 = trans2math(clist, x_[0])

                trans2math_keys = decoder.key_cot_decode(fewshot_stage5, q_stage5, args, max_length)
                clist_new = extract_keys(trans2math_keys)
                print(clist_new)

                if len(clist_new) == len(clist):
                    hint_stage4 = generate_hint(clist_new, qlist, location_dict)
                else:
                    hint_stage4 = generate_hint(clist, qlist, location_dict)

                q_stage4 = x + "A:" + hint_stage4 + "\n"
                answer = decoder.key_cot_decode(fewshot_stage4, q_stage4, args, max_length)
                print(q_stage4)
                print("\n")
                print(answer)
                print("\n")
            elif args.method == "ltm_cot":
                q_stage1 = x + "A:"
                sub_q = decoder.key_cot_decode(fewshot_stage1, q_stage1, args, max_length)
                output_line["sub_questions"] = sub_q
                print(sub_q)

                subq_list = extract_question(sub_q)
                print(subq_list)

                q_stage2 = x + "A:Let's break down this question:"
                j = 1
                for q in subq_list:
                    q_stage2 += ' ' + str(j) + '.' + q
                    j += 1
                q_stage2 += '\n'
                answer = decoder.key_cot_decode(fewshot_stage2, q_stage2, args, max_length)
                print(answer)
                print("\n")
            elif args.method == "few_shot_cot":
                q = x + "A:"
                ck, ft = 0, 0
                while ck == 0 and ft < 10:
                    try:
                        if ft > 0:
                            decoder = Decoder()
                        answer = decoder.key_cot_decode(fewshot_stage1, q, args, max_length)
                        ck = 1
                    except:
                        print("The connection disconnects unexpectedly. Retrying to establish connection.")
                        ft += 1
                if ft == 10:
                    sys.exit(-2)

            pred = answer_cleaning(args, answer)

            output_line["pred_ans"] = pred
            #output_line["wrap_que"] = x

            output_json = json.dumps(output_line)
            wp.write(output_json + '\n')

            # Choose the most frequent answer from the list ...
            print("pred : {}".format(pred))
            print("GT : " + y)
            print('*************************')

            # Checking answer ...
            # correct = (np.array([pred]) == np.array([y])).sum().item()
            y = y.replace(",", "")
            if pred == '':
                correct = 0
            if args.dataset.startswith("math"):
                correct = int(is_equiv(pred, y))
            elif float(pred)==float(y):
                correct = 1
            else:
                correct = 0
            correct_list.append(correct)
            total += 1  # np.array([y]).size(0)

            accuracy = (sum(correct_list) * 1.0 / total) * 100
            print("correct number : {}".format(sum(correct_list) * 1.0))
            print("accuracy : {}".format(accuracy))

            if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
                break
                # raise ValueError("Stop !!")

            # Calculate accuracy ...
        #accuracy = (sum(correct_list) * 1.0 / total) * 100
        #print("accuracy : {}".format(accuracy))

if __name__ == "__main__":
    main()

