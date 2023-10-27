import torch
import gradio as gr
# from model.blip2_opt import Blip2OPT
from stage2 import get_args
from model.blip2_stage2 import Blip2Stage2
from model.blip2_opt import smiles2data
from torch_geometric.loader.dataloader import Collater
from data_provider.stage2_dm import smiles_handler


@torch.no_grad()
def molecule_caption(smiles, prompt, temperature):
    if args.test_ui:
        return f'test {smiles}, {prompt}, {temperature}'
    temperature /= 100
    
    ## process graph prompt
    try:
        graph_batch = collater([smiles2data(smiles)]).to(args.devices)
    except:
        raise gr.Error("The input SMILES is invalid!")
    
    ## process smiles prompt
    prompt = '[START_I_SMILES]{}[END_I_SMILES]. '.format(smiles[:256]) + prompt
    prompt = smiles_handler(prompt, '<mol>' * 8, True)[0]
    molca.opt_tokenizer.padding_side = 'left'
    prompt_batch = molca.opt_tokenizer([prompt, ],
                                       truncation=True,
                                       padding='longest',
                                       add_special_tokens=True,
                                       max_length=384,
                                       return_tensors='pt',
                                       return_attention_mask=True).to(args.devices)

    is_mol_token = prompt_batch.input_ids == molca.opt_tokenizer.mol_token_id
    prompt_batch['is_mol_token'] = is_mol_token
    
    samples = {'graphs': graph_batch, 'prompt_tokens': prompt_batch}
    
    ## generate results
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        text = molca.generate(samples, temperature=temperature, max_length=256, num_beams=2)[0]
    return text



if __name__ == '__main__':
    args = get_args()
    args.devices = f'cuda:{args.devices}'
    args.test_ui = False

    if not args.test_ui:
        # load model
        collater = Collater([], [])
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        molca = model.blip2opt

        del model
        molca = molca.half().eval().to(args.devices)

    with gr.Blocks() as demo:
        gr.HTML(
        """
        <center><h1><b>MolCA</b></h1></center>
        <p style="font-size:20px; font-weight:bold;">This is the demo page of <i>MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter</i>. In EMNLP 2023.</p>
        <center><img src="/file=./figures/finetune.jpg" alt="MolCA Image" style="width:1000px;"></center>
        <p style="font-size:20px; font-weight:bold;"> You can input one smiles below, and we will generate the molecule's text descriptions. </p>
        """)
        smiles = gr.Textbox(placeholder="Input one SMILES", label='Input SMILES')
        prompt = gr.Textbox(placeholder="Customized your own prompt. Note this can give unpredictable results given our model was not pretrained for other prompts.", label='Customized prompt (Default to None)', value='')
        temperature = gr.Slider(0, 100, value=100, label='Temperature')
        btn = gr.Button("Submit")
        out = gr.Textbox(label='Output', placeholder='Molecule caption results')
        btn.click(fn=molecule_caption, inputs=[smiles, prompt, temperature], outputs=out)
    demo.launch(share=True)

