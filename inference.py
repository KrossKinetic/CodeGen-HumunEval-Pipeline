from transformers import pipeline
from datasets import load_dataset
import evaluate
import subprocess

# Global Arrays for Models and Metrics
models = ["meta-llama/Llama-3.1-8B"] 
#,"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B","Qwen/Qwen2.5-Coder-7B-Instruct"
metrics = ["bleu","exact_match"]
code_eval = evaluate.load(metrics[0])

# Loading the primary dataset
ds = load_dataset("THUDM/humaneval-x", "python")["test"]

def init_model_pipe(models):
    pipe_list = []
    for i in models:
        pipe_list.append(pipeline("text-generation", model=i, device=0))
    return pipe_list

def get_message_template(model_name,prompt):
    instructions = "Give me the complete function based on the specification. Only give me one complete function, no comment or explanation is required. Don't repeat functions.:\n" + prompt
    if (model_name == "meta-llama/Llama-3.1-8B"):
        return instructions
    elif (model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        return [{"role": "user", "content": instructions}]
    elif (model_name == "Qwen/Qwen2.5-Coder-7B-Instruct"):
        return [{"role": "user", "content": instructions}]
    else:
        raise ValueError("model_name not in the list")

def run_inference(pipes,dataset):
    all_prediction = []
    all_testcases = []
    
    for pipe in pipes:
        predictions = []
        test_cases = []
        count = 0

        for example in dataset:
            count += 1
            prompt = example["prompt"]

            message = get_message_template(pipe.model.name_or_path,prompt)

            print("Loading... " + str(count) + "/5")
            
            response = pipe(message,max_new_tokens=512)
            generated_code = extract_fn(response[0]['generated_text'])
            print(generated_code[158::])
            predictions.append(generated_code[158::])
            test_cases.append(example["test"])
            
            if count == 50:
                break

        all_prediction.append(predictions)
        all_testcases.append(test_cases)
    
    return (all_prediction,all_testcases)

def run_code(code, timeout=5):
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0  # return True if no errors, False otherwise
    except Exception as e:
        return False # return false for anything else

def pass_at_1(prediction, test_cases):
    combined_code = prediction + "\n" + test_cases
    pass_1 = 0
    if run_code(combined_code):
        pass_1 = 1
    return pass_1


def pass_at_1(prediction, test_cases):
    combined_code = prediction + "\n" + test_cases
    pass_1 = 0
    if run_code(combined_code):
        pass_1 = 1
    return pass_1
    
def extract_fn(generated_code: str) -> str:
    function_definitions = generated_code.split("def ")[1:]
    if len(function_definitions) <= 1:
        return generated_code
    second_function_start_index = generated_code.find("def " + function_definitions[1])
    return generated_code[:second_function_start_index].strip()


list_of_pipes = init_model_pipe(models)
predictions,test_cases = run_inference(list_of_pipes,ds)
total_score = 0

print("Queried successfully")

for prediction,test_case in zip(predictions[0],test_cases[0]):
    print("=================")
    print(prediction)
    print(test_case)
    val = pass_at_1(prediction, test_case)
    print(f"pass@1: {val}")
    total_score += int(val)

print(f"Final score: {total_score/50.0}")