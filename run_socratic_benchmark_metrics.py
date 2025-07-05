from metrics.metric_computer import MetricComputer
from data_utils.data_reader import XMLDataReader
from inference.gpt_inference import ChatGPTModel
from inference.llama_inference import OllamaModel
from data_utils.data_reader import extract_tag_content
import argparse
from tqdm import tqdm
import os
import regex as re
from prompts import *

def debug():
    # metric = MetricComputer(export_to_excel=True, export_path="results.xlsx")

    # predictions = ["hello there, I am badeed", "general kenobi kappacino frank"]
    # references = [
    #     ["hello there, I am badeed", "hi here! you like cats?"],
    #     ["general kenobi kappacino frank", "general grievous badeed neice"]
    # ]

    # scores = metric.compute_overall_score(predictions, references)
    # print(scores)
    dialogue_text = """
User: I am trying to run the code but it is not working.
    <alt> Nothing is working.
Assistant: What is the error message?
    <alt> Any error message?
    <alt> What does the terminal say?
User: I figured it out. I was missing a comma.
<code>
import pandas as pd
if True:
    df = pd.read_csv('data.csv' , sep=',')
</code>
    <alt> Ah! I think it is because the file name is wrong.
<code>
import pandas as pd
if True:
    df = pd.read_csv('my_data.csv')
</code>
Assistant: Great! It is working now.
    <alt> Awesome! Good job!
User: Thank you for your help.
    <alt> Thanks!
"""
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "reviewing_dialogues")
    path = os.path.join(path, "final_dataset")
    data_reader = XMLDataReader(path)
    print(data_reader._parse_dialogue(dialogue_text))


def main(use_chat_prompt, generation_mode, num_responses, dataset_path, eval_method, quality):
    """
    Main function to run the Socratic Benchmark Metrics.

    Args:
        use_chat_prompt (bool): Flag to determine if chat prompt should be used.
        generation_mode (str): Mode of generation, can be "sample", "multiple", or "cot".
        num_responses (int): Number of responses to generate.
        dataset_path (str): Path to the dataset.
        eval_method (str): Evaluation method to use, can be "thoroughness" or other methods.

    Raises:
        ValueError: If an unsupported generation mode is used with chat prompt.

    Returns:
        None
    """
    path = os.path.dirname(os.path.abspath(__file__))
    # split dataset_path by / and replace with os.path.join
    dataset_path = os.path.join(*dataset_path.split("/"))
    path = os.path.join(path, dataset_path)
    data_reader = XMLDataReader(path)
    if generation_mode == "sample" and num_responses > 1:
        n = num_responses
    else:
        n = 1

    parsed_dialogues = data_reader.get_parsed_dialogues()
    api_key = os.environ["OPENAI_API_KEY"]
    steering_prompt = ""
    if generation_mode != "sample":
        generation_args = {
            "max_tokens": 1024,
            "temperature": 0.0,
            "top_p": 0.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "n": n,  # number of responses to return,
            "stream": False,
        }
    else:
        generation_args = {
            "max_tokens": 1024,
            "temperature": 0.75,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "n": n,  # number of responses to return,
            "stream": False,
        }
        
    if quality == "good":
        match generation_mode:
            case "multiple":
                steering_prompt = MULTIPLE_STEERING_PROMPT
            case "cot":
                steering_prompt = COT_STEERING_PROMPT
            case _:
                steering_prompt = STEERING_PROMPT
    elif quality == "bad":
        match generation_mode:
            case "multiple":
                steering_prompt = BAD_MULTIPLE_STEERING_PROMPT
            case "cot":
                steering_prompt = BAD_COT_STEERING_PROMPT
            case _:
                steering_prompt = BAD_STEERING_PROMPT

    chatgpt = ChatGPTModel(
        api_key=api_key,
        generation_args=generation_args,
        steering_prompt=steering_prompt,
    )
    gpt4 = ChatGPTModel(
        model_name="gpt-4",
        api_key=api_key,
        generation_args=generation_args,
        steering_prompt=steering_prompt,
    )
    llama = OllamaModel(
        model_name = "llama3.1-70b",
        steering_prompt=steering_prompt,
        generation_args=generation_args
    )

    # make sure the results folder exists
    if quality == "good":
        results_folder = "results/good"
    elif quality == "bad":
        results_folder = "results/bad"
    else:
        results_folder = "results/"
        
    if not os.path.exists(results_folder + "/gpt4"):
        os.makedirs(results_folder + "/gpt4")
    if not os.path.exists(results_folder + "/chatgpt"):
        os.makedirs(results_folder + "/chatgpt")
    if not os.path.exists(results_folder + "/llama"):
        os.makedirs(results_folder + "/llama")

    if use_chat_prompt:
        chatgpt_path = results_folder + f"/chatgpt/chat_prompt_{generation_mode}_{n}_responses"
        gpt4_path = results_folder + f"/gpt4/chat_prompt_{generation_mode}_{n}_responses"
        llama_path = results_folder + f"/llama/chat_prompt_{generation_mode}_{n}_responses"
    else:
        if generation_mode == "multiple" or generation_mode == "cot":
            chatgpt_path = results_folder + f"/chatgpt/comprehensive_prompt_{generation_mode}_{n}_responses"
            gpt4_path = results_folder + f"/gpt4/comprehensive_prompt_{generation_mode}_{n}_responses"
            llama_path = results_folder + f"/llama/comprehensive_prompt_{generation_mode}_{n}_responses"
        else:
            chatgpt_path = results_folder + f"/chatgpt/instruction_prompt_{generation_mode}_{n}_responses"
            gpt4_path = results_folder + f"/gpt4/instruction_prompt_{generation_mode}_{n}_responses"
            llama_path = results_folder + f"/llama/instruction_prompt_{generation_mode}_{n}_responses"
    if 'java' in dataset_path:
        chatgpt_path = chatgpt_path + "_java.xlsx"
        gpt4_path = gpt4_path + "_java.xlsx"
        llama_path = llama_path + "_java.xlsx"
    elif 'python' in dataset_path:
        chatgpt_path = chatgpt_path + "_python.xlsx"
        gpt4_path = gpt4_path + "_python.xlsx"
        llama_path = llama_path + "_python.xlsx"
    else:
        chatgpt_path = chatgpt_path + ".xlsx"
        gpt4_path = gpt4_path + ".xlsx"
        llama_path = llama_path + ".xlsx"
        
    chatgpt_metrics = MetricComputer(export_to_excel=True, export_path=chatgpt_path)
    gpt4_metrics = MetricComputer(export_to_excel=True, export_path=gpt4_path)
    llama_metrics = MetricComputer(export_to_excel=True, export_path=llama_path)
    
    chatgpt_responses = []
    gpt4_responses = []
    llama_responses = []
    dataset_references = []
    prompts = []
    
    for parsed_dialogue_object in tqdm(parsed_dialogues, desc="Processing dialogues"):
        dialogue = parsed_dialogue_object["dialogue"]
        context_dict = parsed_dialogue_object["context"]
        context_text = context_dict["context_text"]

        if generation_mode == "multiple":

            instruction = "Respond to the user with all possible distinct Socratic utterances that guide the user to discover and fix the bug described between `<bug_desc>` and `</bug_desc>`. Student code is written between `<code>` and `</code>` throughout the conversation. Utterances that have the same meaning but different words are considered duplicates. Assume that the student has run the test cases.\n1."

        elif generation_mode == "cot":

            instruction = {
                "first_introspection": "After looking at the buggy code, the bug description, and the bug fix, what are all the possible reasons or misconceptions that led the user to make this mistake? Do NOT list Socratic questions.",
                "introspection": 'Given the dialogue so far, what are all the possible reasons or misconceptions if any that the user still has that impede them from fixing the bug? Do NOT list Socratic questions. If the bug is already fixed, say "There are no remaining misconceptions or reasons that impede the user from fixing the bug, as they have already identified and corrected their code."',
                "respond": "Utilizing the dialogue and reasons or misconceptions so far, respond to the user with all possible distinct Socratic utterances that guide the user to discover and fix the bug described between `<bug_desc>` and `</bug_desc>`. Student code is written between `<code>` and `</code>` throughout the conversation. Utterances that have the same meaning but different words are considered duplicates. Assume that the student has run the test cases.\n1.",
            }

        else:
            instruction = "Respond to the user with a Socratic utterance that guides the user to discover and fix the bug described between `<bug_desc>` and `</bug_desc>`. Student code is written between `<code>` and `</code>` throughout the conversation. Assume that the student has run the test cases."

        for i, turn in enumerate(dialogue):
            prompt = context_text + "\n\n"
            # dialogue is composed of a list of dictionaries, each dictionary is a turn
            # each turn has the following keys:
            # user_text: the text the user said
            # code_text: the code if any the user has written in this turn
            # cumulative_code_text: the most current code if any the user has written up to this point
            # assistant_text: the text the assistant said
            # user_alternatives: [{alt_text: the alternatives the user could have said, alt_code: any code the user could have written}]
            # assistant_alternatives: the alternatives the assistant could have said.

            # break if the turn has no assistant text
            if i < len(dialogue) and dialogue[i]["assistant_text"] == "":
                break

            if use_chat_prompt:
                if generation_mode == "multiple":
                    raise ValueError(
                        "Multiple generation mode is not supported with chat prompt"
                    )
                elif generation_mode == "cot":
                    raise ValueError(
                        "COT generation mode is not supported with chat prompt"
                    )
                else:
                    # Valid chat prompt generation mode
                    # loop through all previous turns up to this point and add them to a messages list of tuples containing (speaker, text).
                    # then use `generate_turn(messages, user_identifier='User', system_identifier='Assistant')'`
                    messages = []
                    # instantiate messages with the context text and the first turn
                    messages.append(
                        ("User", context_text + "\n\n" + dialogue[0]["user_text"])
                    )

                    for j in range(1, i):
                        prev_turn = dialogue[j]

                        if prev_turn["code_text"] != "":
                            messages.append(
                                (
                                    "User",
                                    prev_turn["user_text"]
                                    + "\n<code>\n"
                                    + prev_turn["code_text"]
                                    + "\n</code>\n",
                                )
                            )
                        else:
                            messages.append(("User", prev_turn["code_text"]))
                        messages.append(("Assistant", prev_turn["assistant_text"]))
                        if prev_turn["code_text"] != "":
                            messages.append(("Assistant", prev_turn["code_text"]))

                        prompt += f"User: {prev_turn['user_text']}\n"
                        prompt += f"Assistant: {prev_turn['assistant_text']}\n"
                        if prev_turn["code_text"] != "":
                            prompt += f"<code>\n{prev_turn['code_text']}\n</code>\n"

                    # add the current turn to the messages list
                    if turn["code_text"] != "" and i > 0:
                        messages.append(
                            (
                                "User",
                                turn["user_text"]
                                + "\n<code>\n"
                                + turn["code_text"]
                                + "\n</code>\n",
                            )
                        )
                    elif i > 0:
                        messages.append(("User", turn["user_text"]))

                    # finish the prompt with the current turn
                    prompt += f"User: {turn['user_text']}\n"
                    if turn["code_text"] != "":
                        prompt += f"<code>\n{turn['code_text']}\n</code>\n"

                    chatgpt_response = chatgpt.generate_turn(
                        messages, user_identifier="User", system_identifier="Assistant"
                    )
                    gpt4_response = gpt4.generate_turn(
                        messages, user_identifier="User", system_identifier="Assistant"
                    )
                    llama_response = llama.generate_turn(
                        messages, user_identifier="User", system_identifier="Assistant"
                    )
                    prompts.append(prompt)

            else:
                # if in instruction prompt mode, add the instruction to the prompt
                # loop through all previous turns up to this point and add them to the prompt
                if i == 0 and generation_mode == "cot":
                    introspection_prompt = (
                        prompt + f"{instruction['first_introspection']}\n"
                    )
                    # generate responses
                    introspection_chatgpt = chatgpt.generate(introspection_prompt)
                    introspection_gpt4 = gpt4.generate(introspection_prompt)
                    introspection_llama = llama.generate(introspection_prompt)

                for j in range(i):
                    if generation_mode == "cot" and j == 0:
                        prompt += "\nDialogue:\n"

                    prev_turn = dialogue[j]
                    prompt += f"User: {prev_turn['user_text']}\n"
                    prompt += f"Assistant: {prev_turn['assistant_text']}\n"
                    if prev_turn["code_text"] != "":
                        prompt += f"<code>\n{prev_turn['code_text']}\n</code>\n"

                prompt += f"User: {turn['user_text']}\n"
                if turn["code_text"] != "":
                    prompt += f"<code>\n{turn['code_text']}\n</code>\n"

                if generation_mode == "cot":
                    if 1 > i and generation_mode == "cot":
                        introspection_prompt = (
                            prompt + f"{instruction['introspection']}\n"
                        )
                        # generate responses
                        introspection_chatgpt = chatgpt.generate(introspection_prompt)
                        introspection_gpt4 = gpt4.generate(introspection_prompt)
                        introspection_llama = llama.generate(introspection_prompt)

                    gpt4_prompt = (
                        prompt
                        + f"\nReasons or Misconceptions so far:\n{introspection_gpt4}\n{instruction['respond']}"
                    )
                    chatgpt_prompt = (
                        prompt
                        + f"\nReasons or Misconceptions so far:\n{introspection_chatgpt}\n{instruction['respond']}"
                    )
                    llama_prompt = (
                        prompt
                        + f"\nReasons or Misconceptions so far:\n{introspection_llama}\n{instruction['respond']}"
                    )
                    save_prompt = (
                        prompt
                        + f"\nReasons or Misconceptions so far:\n\tchatgpt:\n{introspection_gpt4}\n\tgpt4:\n{introspection_gpt4}\n\tllama:\n{introspection_llama}\n{instruction['respond']}"
                    )
                    # generate responses
                    chatgpt_response = chatgpt.generate(chatgpt_prompt)
                    gpt4_response = gpt4.generate(gpt4_prompt)
                    llama_response = llama.generate(llama_prompt)
                    prompts.append(save_prompt)

                else:
                    prompt += f"\n{instruction}"

                    # generate responses
                    chatgpt_response = chatgpt.generate(prompt)
                    gpt4_response = gpt4.generate(prompt)
                    llama_response = llama.generate(prompt)
                    prompts.append(prompt)

            references = [turn["assistant_text"]] + turn["assistant_alternatives"]
            dataset_references.append(references)

            if generation_mode == "multiple" or generation_mode == "cot":
                # parse the itemized list string response into a list of responses
                if not chatgpt_response.startswith("1.") and not chatgpt_response.startswith("1. "):
                    chatgpt_response = (
                        "1." + chatgpt_response
                        if chatgpt_response.startswith(" ")
                        else "1. " + chatgpt_response
                    )
                if not gpt4_response.startswith("1.") and not gpt4_response.startswith("1. "):
                    gpt4_response = (
                        "1." + gpt4_response
                        if gpt4_response.startswith(" ")
                        else "1. " + gpt4_response
                    )
                if not llama_response.startswith("1.") and not llama_response.startswith("1. "):
                    llama_response = (
                        "1." + llama_response
                        if llama_response.startswith(" ")
                        else "1. " + llama_response
                    )
                # clean responses from any code blocks
                if "<code>" in chatgpt_response:
                    remove_code = extract_tag_content(chatgpt_response, "code")
                    chatgpt_response = chatgpt_response.replace(
                        f"<code>\n{remove_code}\n</code>", ""
                    )
                if "<code>" in gpt4_response:
                    remove_code = extract_tag_content(gpt4_response, "code")
                    gpt4_response = gpt4_response.replace(
                        f"<code>\n{remove_code}\n</code>", ""
                    )
                if "<code>" in llama_response:
                    remove_code = extract_tag_content(llama_response, "code")
                    llama_response = llama_response.replace(
                        f"<code>\n{remove_code}\n</code>", ""
                    )
                # create a list of responses from the itemized list string response while removing the item numbers using regex
                chatgpt_response = re.findall(r"\d+\.\s*(.*)", chatgpt_response)
                gpt4_response = re.findall(r"\d+\.\s*(.*)", gpt4_response)
                llama_response = re.findall(r"\d+\.\s*(.*)", llama_response)

            # Note: if generation mode is multiple or sample with n > 1, then we have multiple responses per sample.
            chatgpt_responses.append(chatgpt_response)
            gpt4_responses.append(gpt4_response)
            llama_responses.append(llama_response)

    # compute metrics for all generation modes.
    if eval_method != "thoroughness":
        chat_gpt_scores = chatgpt_metrics.compute(
            chatgpt_responses, dataset_references, contexts=prompts
        )
        gpt4_scores = gpt4_metrics.compute(
            gpt4_responses, dataset_references, contexts=prompts
        )
        llama_scores = llama_metrics.compute(
            llama_responses, dataset_references, contexts=prompts
        )
    else:
        chat_gpt_scores = chatgpt_metrics.compute_thoroughness(
            chatgpt_responses, dataset_references, contexts=prompts
        )
        gpt4_scores = gpt4_metrics.compute_thoroughness(
            gpt4_responses, dataset_references, contexts=prompts
        )
        llama_scores = llama_metrics.compute_thoroughness(
            llama_responses, dataset_references, contexts=prompts
        )
    print("ChatGPT scores:")
    print(chat_gpt_scores)
    print("GPT-4 scores:")
    print(gpt4_scores)
    print("Llama scores:")
    print(llama_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run debug mode")
    parser.add_argument(
        "--do_chat_prompt",
        action="store_true",
        help="Run chat prompt mode where the prompt is the chat history between the user and chatgpt. Defaults to an instructional prompt 'Respond to the user with a Socratic utterance that guides the user to discover and fix the bug described in `<bug_desc>`. Student code is denoted by `<code>` throughout the conversation. Utterances that have the same meaning but different words are considered duplicates. Assume that the student has run the test cases.'",
    )
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="multiple",
        help="Generate a single response or multiple responses using instruction or sample k responses. Options are 'single', 'multiple', 'cot', or 'sample'. Defaults to 'single'. 'multiple' is incompatible with '--do_chat_prompt'. If 'sample', the number of responses to sample is specified by the --num_responses argument.",
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=5,
        help="Number of responses to sample if --generation_mode is 'sample'. Defaults to 5.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="socratic_debugging_benchmark/testing",
        help="Path to the dataset to evaluate. Defaults to 'reviewing_dialogues/eval_full_v2'.",
    )
    parser.add_argument(
        "--evaluation_method",
        type=str,
        default="thoroughness",
        help="Evaluation method. Either 'overall' or 'thoroughness'. Defaults to 'thoroughness'. Thoroughness uses the bipartite matching algorithm to find the best match between each reference and the response. Overall uses scoring of the best matching (reference, pred) pair.",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="good",
        help="Quality of the responses. Either 'good' or 'bad'. Defaults to 'good'.",
    )
    args = parser.parse_args()
    if args.debug:
        debug()
    else:
        main(
            args.do_chat_prompt,
            args.generation_mode,
            args.num_responses,
            args.dataset_path,
            args.evaluation_method,
            args.quality
        )
        # debug()


# run with:
# instruction mode with a single response
# python run_socratic_benchmark_metrics.py (done on dekstop)
# single response without instruction mode
# python run_socratic_benchmark_metrics.py --do_chat_prompt (ready to test)
# multiple responses (all possible socratic utterances) with instruction mode
# python run_socratic_benchmark_metrics.py --generation_mode multiple (ready to test)
# sample 5 responses with instruction mode
# python run_socratic_benchmark_metrics.py --generation_mode sample --num_responses 5 (ready to test)
# sample 10 responses with instruction mode
# python run_socratic_benchmark_metrics.py --generation_mode sample --num_responses 10
# sample 5 responses without instruction mode
# python run_socratic_benchmark_metrics.py --do_chat_prompt --generation_mode sample --num_responses 5
# sample 10 responses without instruction mode
# python run_socratic_benchmark_metrics.py --do_chat_prompt --generation_mode sample --num_responses 10
