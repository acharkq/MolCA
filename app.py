import torch
import gradio as gr
# from model.blip2_opt import Blip2OPT
from stage2 import get_args
from model.blip2_stage2 import Blip2Stage2
from model.blip2_opt import smiles2data
from torch_geometric.loader.dataloader import Collater
from data_provider.stage2_dm import smiles_handler
from rdkit import Chem
from rdkit.Chem import Draw

@torch.no_grad()
def molecule_caption(smiles, prompt, temperature):
    if args.test_ui:
        mol = Chem.MolFromSmiles(smiles)
        # Define the resolution of the image
        img = Draw.MolToImage(mol, size=(900,900))
        return f'test {smiles}, {prompt}, {temperature}', img
    # temperature /= 100
    
    ## process graph prompt
    try:
        graph_batch = collater([smiles2data(smiles)]).to(args.devices)
    except:
        raise gr.Error("The input SMILES is invalid!")
    
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol,size=(900,900))

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
        text = molca.generate(samples, temperature=temperature, max_length=256, num_beams=2, do_sample=True)[0]
    return text, img



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
        ## list of examples
        example_list = ['CC1=C(SC(=[N+]1CC2=CN=C(N=C2N)C)C(CCC(=O)O)O)CCOP(=O)(O)OP(=O)(O)O', 'CCCCCCCCCCCCCCCC/C=C\\OC[C@H](COP(=O)(O)O)O', 'C1=CC=C(C=C1)[As](=O)(O)[O-]', 'CCCCCCCCCCCC(=O)OC(=O)CCCCCCCCCCC', 'C(C(C(=O)O)NC(=O)N)C(=O)O', 'CCCCCCCCCCCCCCCC(=O)OC(CCCCCCCCC)CCCCCCCC(=O)O', 'CC1=CC(=O)C(=C(C1=O)O)OC']
        gr.Examples(example_list, [smiles,], fn=molecule_caption, label='Example SMILES')

        prompt = gr.Textbox(placeholder="Customized your own prompt. Note this can give unpredictable results given our model was not pretrained for other prompts.", label='Customized prompt (Default to None)', value='')
        temperature = gr.Slider(0.1, 1, value=1, label='Temperature')
        btn = gr.Button("Submit")

        with gr.Row():
            out = gr.Textbox(label='Caption Output', placeholder='Molecule caption results')
            img_out = gr.Image(label='Molecule 2D Structure', placeholder="Visualizing the Molecule's 2D structures.")
        btn.click(fn=molecule_caption, inputs=[smiles, prompt, temperature], outputs=[out, img_out])
    demo.launch(share=True)

