#!/bin/bash
#SBATCH --job-name=paraprhase_llm
#SBATCH --time=72:00:00
#SBATCH -N 1

export OPENAI_API_KEY=""
export TGI_SERVER_URL=""

source ~/.bashrc
conda activate nlp

# Your Discord webhook URL
webhook_url=""

# The message you want to send
message="A new job started."

# JSON payload
json_payload=$(cat <<EOF
{
    "username": "TGI API Bot",
    "avatar_url": "https://aeiljuispo.cloudimg.io/v7/https://cdn-uploads.huggingface.co/production/uploads/5f17f0a0925b9863e28ad517/NXI_YNqaf9ZvhVczj0kpz.png?w=200&h=200&f=face",
    "content": "$message",
    "embeds": [
        {
            "title": "Job Scheduler",
            "description": "A new job started on $(hostname)",
            "color": 15258703
        }
    ]
}
EOF
)

# Use curl to send a POST request
curl -X POST \
    -H "Content-Type: application/json" \
    -d "$json_payload" \
    $webhook_url


# run the script
python3 src/paraphrase_and_run_tasks.py
