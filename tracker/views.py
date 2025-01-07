from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openprompt.plms import load_plm
from openprompt.prompts.prompt_generator import LMBFFTemplateGenerationTemplate
from openprompt.pipeline_base import PromptDataLoader, PromptForClassification
from openprompt.prompts import ManualVerbalizer, ManualTemplate
from openprompt.data_utils import InputExample
import pandas as pd
import torch
from torch import nn
import random
import copy

# Load the pre-trained model and tokenizer
plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-large")

# number of demonstrations
num_demonstrations = 1  # try different number

demonstrations = []

for _ in range(num_demonstrations):
    random_example_1 = "how about this and a reference to the rocky mountain chronicle as mentioned betting here there are circ several other third party sources to the climate bets we have scam reporting that gray has offered to bet as well as the businessandmedia link"
    random_example_2 = "if you can find any biological proof this turns out man gives little idea by all where women came from so if one breed with one you get a good meat special"

    demonstration = f'{random_example_1} Sentence for analyze: This sentence is positive.'\
                    f' '\
                    f'{random_example_2} Sentence for analyze: This sentence is negative.'
    demonstrations.append(demonstration)


# Prepare the template and verbalizer
template = ManualTemplate(
    tokenizer=tokenizer, 
    text='{"placeholder":"text_a"} Sentence for analyze: This sentence is {"mask"}.' 
        + ' '.join(demonstrations)
)

classes = [ 
    "negative",
    "positive"
]
verbalizer = ManualVerbalizer(
    tokenizer=tokenizer, 
    classes = classes,
    num_classes=2, 
     label_words = {
        "positive": ["positive", "good"],
        "negative": ["negative", "Hindu mob"],
         
    }
)

# Define the model for classification
model = PromptForClassification(
    copy.deepcopy(plm),
    template,
    verbalizer
)


# Load the trained model
# model.load_state_dict(torch.load('best_model_by_template_01_04_25.pt'))

# Function to convert a DataFrame to InputExamples
def df_to_inputexamples(df):
    input_examples = []
    for _, row in df.iterrows():
        text = row['Content']
        label = row['Label']
        input_example = InputExample(text_a=text, label=label)
        input_examples.append(input_example)
    return input_examples

# Define an API view to classify the input text
@csrf_exempt
def classify_text(request):
    if request.method == "POST":
        try:
            # Get text input from the request
            input_text = request.POST.get('text', None)
            if not input_text:
                return JsonResponse({'error': 'No text provided'}, status=400)

            # Convert the input text into an InputExample
            # input_example = InputExample(text_a=input_text, label=None)
            # #test_input_examples = [input_example, 1]

            # test_df = pd.DataFrame(columns=['Content', 'Label'])
            # test_df.loc[len(test_df)] = [input_text, 1]
            # print(test_df)

            # test_input_examples = df_to_inputexamples(test_df)

            # # Create PromptDataLoader
            # test_dataloader = PromptDataLoader(
            #     dataset=test_input_examples,
            #     template=template,
            #     tokenizer=tokenizer,
            #     tokenizer_wrapper_class=WrapperClass,
            #     decoder_max_length=256,
            #     max_seq_length=256,
            #     batch_size=16
            # )

            # # Perform the classification
            # model.eval()
            # with torch.no_grad():
            #     for inputs in test_dataloader:
            #         inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            #         logits = model(batch=inputs)
            #         pred = torch.argmax(logits, dim=1).item()

            # # Map prediction to label
            # label = classes[pred]
            label = 1
            return JsonResponse({'text': input_text, 'prediction': label})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid HTTP method'}, status=405)

