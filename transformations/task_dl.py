"""Task manager
This script will add the tasks in the queue
"""
#!/usr/bin/env python
import argparse
import os
import pika
import re

parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
args = parser.parse_args()


def put_file_queue(ch, file_name):
    """Add files to the queue

    Args:
        ch (_type_): channel of the queue
        file_name (string): name of file
    """
    ch.basic_publish(
        exchange='',
        routing_key='task_queue_deepl',
        body=file_name,
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        ))


credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', credentials=credentials, heartbeat=5))
channel = connection.channel()

channel.exchange_declare(exchange='dlx', exchange_type='direct')

channel.queue_declare(queue='task_queue_deepl', durable=True, arguments={"dead-letter-exchange": "dlx"})
dl_queue = channel.queue_declare(queue='dl')

channel.queue_bind(exchange='dlx', routing_key='task_queue_deepl', queue=dl_queue.method.queue)

epochs = [100, 200]
batch_size = [50, 100]
embedding_dim = [32, 64]

for file in os.listdir(args.input_folder):
    if  file.split('.')[0].split('_')[0] not in ['loans', 'students', 'diabets', 'lawschool', 'dutch']:
        print(file)
        for eps in epochs:
            for bsz in batch_size:
                for ebd in embedding_dim:
                    put_file_queue(channel, f'{file.split(".")[0]}_CTGAN_{eps}_{bsz}_{ebd}')
                    put_file_queue(channel, f'{file.split(".")[0]}_TVAE_{eps}_{bsz}_{ebd}')
                    put_file_queue(channel, f'{file.split(".")[0]}_CopulaGAN_{eps}_{bsz}_{ebd}')

connection.close()
