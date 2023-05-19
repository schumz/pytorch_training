
def save_model(model:nn.Module,nom_modele):

    models_path=Path("D:\Adam\Pytorch\exercises")
    nom_du_modele = nom_modele.pt

    model_path=models_path/nom_du_modele

    torch.save(obj=model.state_dict(),f=model_path)
