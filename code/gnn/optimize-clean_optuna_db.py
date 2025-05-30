#####
# CLEAN OPTUNA DB #
#####
import optuna

db_name_from = "optuna_study-layered.db"
db_name_to = "optuna_study.db"
studies = ["GCN-export-l-", "GAT-export-l", "SAGE-export-l"]
action = "copy" # "copy" or "delete"

# Define the storage file (saved locally)
from_storage = optuna.storages.RDBStorage(f"sqlite:///{db_name_from}", engine_kwargs={"connect_args": {"timeout": 30}})
to_storage = optuna.storages.RDBStorage(f"sqlite:///{db_name_to}", engine_kwargs={"connect_args": {"timeout": 30}})


for study_name in studies:
    if action == "delete":
        optuna.delete_study(study_name=study_name, storage=from_storage)
    elif action == "copy":
        print(f"Copying study {study_name}")
        optuna.copy_study(
            from_study_name=study_name,
            from_storage=from_storage,
            to_storage=to_storage,
        )