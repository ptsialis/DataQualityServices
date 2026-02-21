#Copyright 2024 Mücahit Sahin
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import json
import uuid
import time
import openai


def load_api_key():
    with open('OpenAIAPIKey.txt') as file:
        openai.api_key = file.read()
        file.close()

def add_batch(batchfile=None, messages=[], model="gpt-4o"):
    if batchfile == None:
        return 'Specify the batchfile to add the batch!'
    batch = {
        'custom_id': 'request-'+str(uuid.uuid4()),
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': {
            'model': model,
            'messages': messages,
            'temperature':0.3, #1.0, #0.3, 0.7
            'seed': 42
        }
    }
    if os.path.isfile(batchfile):
        with open(batchfile, 'a') as fp:
            fp.write('\n')
            json.dump(batch, fp)
            fp.close()
    else:
        with open(batchfile, 'w') as fp:
            json.dump(batch, fp)
            fp.close()


def ask_gpt(prompt=None, system_prompt=None, model="gpt-4o", batchfile=None):
    client = openai.OpenAI(api_key=openai.api_key)

    messages = [
        {"role": "user", "content": prompt}
    ]

    if not system_prompt == None:
        messages.insert(0, {"role": "system", "content": system_prompt})

    if not batchfile == None:
        try:
            batch_input_file = client.files.create(
                file=open(batchfile, "rb"),
                purpose="batch"
            )
            batch_input_file_id = batch_input_file.id
            created_batch = client.batches.create(input_file_id=batch_input_file_id,
                                                  endpoint="/v1/chat/completions",
                                                  completion_window='24h'
                                                 )
            print(f'Batch was created with id: {created_batch.id}\n')
            print(f'-----------------------------------------------------------------\n')
            current_batch = client.batches.retrieve(created_batch.id)
            start = time.time()
            print(f'\rStatus: {current_batch.status}', end="", flush=True)
            try:
                while not current_batch.status == 'completed':
                    end = time.time()
                    time_diff = end-start
                    hours = time_diff//3600
                    time_diff = time_diff - 3600*hours
                    minutes = time_diff//60
                    seconds = time_diff - 60*minutes
                    elapsed_time = f'Elapsed Time: {int(hours)} Hours {int(minutes)} Minutes {int(seconds)} Seconds'
                    print(f'\rStatus: {current_batch.status}, {elapsed_time}', end="", flush=True)
                    if current_batch.status == 'failed' or current_batch.status == 'expired' or current_batch.status == 'cancelled':
                        break
                    else:
                        time.sleep(20)
                        current_batch = client.batches.retrieve(created_batch.id) 
                if current_batch.status == 'expired' or current_batch.status == 'cancelled':
                    print(f'\rStatus: {current_batch.status}', end="", flush=True)
                    print(f'\nOutputfile: {current_batch.output_file_id}')
                    response = client.files.content(current_batch.output_file_id).text
                    print(f'\nWriting to file...')
                    with open(batchfile.replace('.jsonl', '') + '_results.jsonl', 'w') as fp:
                        fp.write(response)
                        fp.close()
                    with open(batchfile.replace('.jsonl', '') + '.log', 'w') as fp:
                        fp.write(elapsed_time)
                        fp.close()
                    print('Done!')
                    return
                elif current_batch.status == 'completed':
                    print(f'\rStatus: {current_batch.status}', end="", flush=True)
                    print(f'\nOutputfile: {current_batch.output_file_id}')
                    response = client.files.content(current_batch.output_file_id).text
                    print(f'\nWriting to file...')
                    with open(batchfile.replace('.jsonl', '') + '_results.jsonl', 'w') as fp:
                        fp.write(response)
                        fp.close()
                    with open(batchfile.replace('.jsonl', '') + '.log', 'w') as fp:
                        fp.write(elapsed_time)
                        fp.close()
                    print('Done!')
                    return [json.loads(r)['response']['body']['choices'][0]['message']['content'].strip() for r in response.split('\n')[:-1]]
                else:
                    print(f'\n\nBatch failed with status: {current_batch.status}')
                    return
            except KeyboardInterrupt:
                print(f'\n\nBatch stopped by user! Cancelling...')
                client.batches.cancel(created_batch.id)
                return
        except Exception as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            return 'Failed to make LLM-prediction'
    else:
        if prompt == None:
            return 'Please specify a user prompt'
        else:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3, #1.0,#0.3, #0.7
                    seed=42
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error: {e} \n")
                return 'Failed to make LLM-prediction'