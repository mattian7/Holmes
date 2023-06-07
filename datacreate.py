import os
import json


def main():
    file_path = 'template/template_multiarith.json'
    write_file = 'dataset/MultiArith-IC/test.json'

    with open(file_path) as f:
        json_data = json.load(f)
        json_data = json_data["demos"]
        res = []
        for line in json_data:
            answer = line["gold_ans"]
            question = line["question"]
            numbers = line["number"][0:3]
            roles = line["role"][0:2]
            insert_index = question.find('[Irrelevant Sentence]')
            in_topic_sentence = []
            in_topic_template = line["in-topic"]
            if in_topic_template[-1]=='.':
                in_topic_template = in_topic_template[0:-1]
            role_index = in_topic_template.find('[ROLE]')
            number_index = in_topic_template.find('[NUMBER]')
            for role in roles:
                for number in numbers:
                    if role_index < number_index:
                        irr_sentence = in_topic_template[:role_index] + role + in_topic_template[role_index+6:number_index] + str(number) + in_topic_template[number_index+8:]
                    else:
                        irr_sentence = in_topic_template[:number_index] + str(number) + in_topic_template[number_index+8:role_index] + role + in_topic_template[role_index+6:]
                    in_topic_sentence.append(irr_sentence)

            off_topic_sentence = []
            off_topic_template = line["off-topic"]
            if off_topic_template[-1]=='.':
                off_topic_template = off_topic_template[0:-1]
            role_index = off_topic_template.find('[ROLE]')
            number_index = off_topic_template.find('[NUMBER]')
            for role in roles:
                for number in numbers:
                    if role_index < number_index:
                        irr_sentence = off_topic_template[:role_index] + role + off_topic_template[role_index + 6:number_index] + str(number) + off_topic_template[number_index + 8:]
                    else:
                        irr_sentence = off_topic_template[:number_index] + str(number) + off_topic_template[number_index + 8:role_index] + role + off_topic_template[role_index + 6:]
                    off_topic_sentence.append(irr_sentence)

            for se in in_topic_sentence:
                index = len(res) + 1
                new_data = {}
                new_q = question[:insert_index] + se + question[insert_index+21:]
                new_data["iIndex"] = index
                new_data["sQuestion"] = new_q
                new_data["lSolutions"] = [answer]
                res.append(new_data)

            for se in off_topic_sentence:
                index = len(res) + 1
                new_data = {}
                new_q = question[:insert_index] + se + question[insert_index+21:]
                new_data["iIndex"] = index
                new_data["sQuestion"] = new_q
                new_data["lSolutions"] = [answer]
                res.append(new_data)

        with open(write_file, 'w') as wf:
            json.dump(res, wf)


if __name__ == "__main__":
    main()