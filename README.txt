- Pour lancer normalement:
	sudo docker run --rm -v <chemin pour data>:/app/data -p 8501:8501 climbing
- Pour lancer avec GPU pour faire le tracking (vitesse x 10):
	Installer nvidia container toolkit
	sudo docker run --rm --gpus all -v <chemin pour data>:/app/data -p 8501:8501 climbing

Seul le dossier <chemin pour data> sera visible pour l'application et sera considéré comme "/app/data". Dans les chemins ou les fichiers de config, TOUS les chemins doivent être spécifié à partir du dossier "/app" donc doivent être du type "./data/videos/myvideo.mp4".


