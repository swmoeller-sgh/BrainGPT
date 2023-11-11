def generate_dictionary(question, answer, sources):
    source_list = []

    for source in sources:
        source_dict = {
            'extract': source['extract'],
            'file_link': source['file_link']
        }
        source_list.append(source_dict)

    data_dict = {
        'question': question,
        'answer': answer,
        'sources': source_list
    }

    return data_dict

# Example usage with multiple questions
questions_and_answers = [
    {
        'question': "What is the capital of France?",
        'answer': "Paris",
        'sources': [
            {
                'extract': "Paris is the capital of France...",
                'file_link': "path/to/source1.txt"
            },
            {
                'extract': "In the heart of Western Europe, Paris stands...",
                'file_link': "path/to/source2.txt"
            }
        ]
    },
    {
        'question': "Who is the author of 'Romeo and Juliet'?",
        'answer': "William Shakespeare",
        'sources': [
            {
                'extract': "William Shakespeare, the renowned playwright...",
                'file_link': "path/to/source3.txt"
            },
            {
                'extract': "One of Shakespeare's most famous works...",
                'file_link': "path/to/source4.txt"
            }
        ]
    }
]

result_list = []

for qa_pair in questions_and_answers:
    result_dict = generate_dictionary(qa_pair['question'], qa_pair['answer'], qa_pair['sources'])
    result_list.append(result_dict)

# Display the generated list of dictionaries
for result in result_list:
    print(result)

print(result_dict.keys())
