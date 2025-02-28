from tqdm import tqdm
import modal
import modal.runner
import pandas as pd
import torch
import torch.nn.functional as F

app = modal.App.lookup(name="recommender", create_if_missing=True)
volume = modal.Volume.from_name("recommender", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.9").pip_install("pandas", "torch")


def batch_infer(df, inputs, batch_size=8, device="cuda"):
    embed_matrix = [df.loc[df.title.isin(movies)].embed.tolist() for movies in inputs]
    embed_matrix = torch.tensor(embed_matrix).to(device)
    embeds = embed_matrix.mean(dim=1)
    rest_embeds = df.loc[~df.title.isin(inputs[0])].embed.tolist()
    rest_embeds = torch.tensor(rest_embeds).to(device)
    sim = []
    for i in tqdm(range(0, len(embeds), batch_size)):
        batch_similarity = batch_similarity = F.cosine_similarity(
            embeds[i : i + batch_size].unsqueeze(1),
            rest_embeds.unsqueeze(0),
            dim=2,
        )
        sim.append(batch_similarity)
    sim = torch.cat(sim)
    idx = torch.topk(sim, 5).indices.cpu().numpy()
    recommendations = []
    for ix, movies in enumerate(inputs):
        recs = df.loc[~df.title.isin(movies)].iloc[idx[ix]].title.tolist()
        recommendations.append(recs)
    return recommendations


def upload_data():
    with volume.batch_upload() as batch:
        batch.put_file("movies_embeds.pkl", "/data/movies_embeds.pkl")
        batch.put_file("infer_dataset_10k.csv", "/data/infer_dataset_10k.csv")
        batch.put_file("infer_dataset_100k.csv", "/data/infer_dataset_100k.csv")


@app.function(
    schedule=modal.Cron("0 17 * * *"),
    volumes={"/volume": volume},
    image=image,
    gpu=["A10G"],
)
def infer():
    volume.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    movie_database = pd.read_pickle("/volume/data/movies_embeds.pkl")
    daily_dataset = pd.read_csv("/volume/data/infer_dataset_100k.csv")
    recommendations = batch_infer(movie_database, daily_dataset.values, device=device)
    recommendations = pd.DataFrame(
        recommendations, columns=["rec1", "rec2", "rec3", "rec4", "rec5"]
    )
    recommendations.to_csv("/volume/data/daily_recommendations_100k.csv", index=False)


if __name__ == "__main__":
    upload_data()
    modal.runner.deploy_app(app)
