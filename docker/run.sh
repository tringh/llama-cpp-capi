docker run -d \
  --gpus all \
  -p 2622:22 \
  -v $SSH_AUTH_SOCK:/ssh-agent \
  -e SSH_AUTH_SOCK:=/ssh-agent \
  -v ~/.gitconfig:/home/developer/.gitconfig \
  -v .:/code/llama-cpp-capi \
  -v ~/mlearn/models:/models \
  --name llama-cpp-capi-dev \
  htring/llamacppdev:0.0.1
