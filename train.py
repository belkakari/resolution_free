import pytorch_lightning as pl

from resolution_free.models import SingleImageModel


config = {'img_path':'/mnt/evo/datasets/wikiart/for_colorconv/giorgione_nymphs-and-children-in-a-landscape-with-shepherds.jpg',
          'max_crop':128,
          'len_dloader':1000,
          'out_folder':'/home/gleb/code/jupyter/coordpred/pl/',
          'layers':[0, 4, 9, 18, 27, 36],
          'vgg_weights':[1, 1, 2, 3, 40],
          'lr':0.001,
          'model':{'internal_dim':20},
          'data':{'batch_size': 5, 'num_workers': 10}}
model = SingleImageModel(config)

trainer = pl.Trainer(gpus=1, show_progress_bar=True)
trainer.fit(model)
