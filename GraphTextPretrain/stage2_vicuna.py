# Need to call this before importing transformers.
from fastchat.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

from stage2 import main, get_args

if __name__ == '__main__':
    main(get_args())