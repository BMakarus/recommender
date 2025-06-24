import argparse
import os
import pickle
from modules.train_functions import train_nmf_model, train_svd_model, train_svd2_model, train_sgd_model
from modules.predict_functions import predict_nmf, predict_svd, predict_svd2, predict_sgd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple Recommender")
    parser.add_argument("--train", type=str, default="no",
                        help="Train mode: 'yes' to train the models, 'no' otherwise.")
    parser.add_argument("--predict", type=str, default="no",
                        help="Predict mode: 'yes' to run prediction, 'no' otherwise.")
    parser.add_argument("--train_file", type=str, default="data/ratings.csv",
                        help="CSV file with training data (userId,movieId,rating).")
    parser.add_argument("--input_file", type=str, default="data/sample_test.csv",
                        help="CSV file with (userId,movieId) for predictions.")
    parser.add_argument("--model_path", type=str, default="models_trained/model.pkl",
                        help="Path to save/load the trained model.")
    parser.add_argument("--output_file", type=str, default="results/preds.csv",
                        help="Where to save predictions.")
    parser.add_argument("--alg", type=str, default="NMF",
                        help="Algorithm to use ('NMF', 'SVD1', 'SVD2', 'SGD','ALL').")
    return parser.parse_args()


def main():
    args = parse_arguments()
    train_mode = (args.train.lower() == "yes")
    predict_mode = (args.predict.lower() == "yes")

    if train_mode:
        if (args.alg == "NMF"):
            print("Training mode for NMF activated.")
            Z_approx, user_map, movie_map = train_nmf_model(args.train_file)
            model_data = {
                "Z_approx": Z_approx,
                "user_map": user_map,
                "movie_map": movie_map
            }
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            with open(args.model_path, "wb") as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {args.model_path}")
        if (args.alg == "SVD1"):
            print("Training mode for SVD1 activated.")
            Z_approx, user_map, movie_map = train_svd_model(args.train_file)
            model_data = {
                "Z_approx": Z_approx,
                "user_map": user_map,
                "movie_map": movie_map
            }
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            with open(args.model_path, "wb") as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {args.model_path}")
        if (args.alg == "SGD"):
            print("Training mode for SGD activated.")
            W, H, user_map, movie_map = train_sgd_model(args.train_file)
            model_data = {
                "W": W,
                "H": H,
                "user_map": user_map,
                "movie_map": movie_map
            }
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            with open(args.model_path, "wb") as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {args.model_path}")
        if (args.alg == "ALL"):
            print("Training mode for all activated.")
            Z_approx_nmf, user_map, movie_map = train_nmf_model(args.train_file)
            Z_approx_svd = train_svd_model(args.train_file)[0]
            Z_approx_svd2 = train_svd2_model(args.train_file)[0]
            W, H = train_sgd_model(args.train_file)[:2]
            models_data = {
                "Z_approx": Z_approx_nmf,
                "user_map": user_map,
                "movie_map": movie_map
            },{
                "Z_approx": Z_approx_svd,
                "user_map": user_map,
                "movie_map": movie_map
            },{
                "Z_approx": Z_approx_svd2,
                "user_map": user_map,
                "movie_map": movie_map
            },{"W": W,
               "H": H,
               "user_map": user_map,
               "movie_map": movie_map
            }
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            with open(args.model_path, "wb") as f:
                pickle.dump(models_data, f)
            print(f"Model saved to {args.model_path}")
    if predict_mode:
        if (args.alg == "NMF"):
            print("Prediction mode activated for NMF.")
            if not os.path.exists(args.model_path):
                print("Model file does not exist. Please run training first.")
                return
            with open(args.model_path, "rb") as f:
                model_data = pickle.load(f)
            predictions = predict_nmf(args.input_file, model_data)
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, "w") as f:
                f.write("userId,movieId,rating\n")
                for row in predictions:
                    f.write(f"{row['userId']},{row['movieId']},{row['rating']}\n")
            print(f"Predictions saved to {args.output_file}")
        if (args.alg == "SVD1"):
            print("Prediction mode activated for SVD1.")
            if not os.path.exists(args.model_path):
                print("Model file does not exist. Please run training first.")
                return
            with open(args.model_path, "rb") as f:
                model_data = pickle.load(f)
            predictions = predict_svd(args.input_file, model_data)
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, "w") as f:
                f.write("userId,movieId,rating\n")
                for row in predictions:
                    f.write(f"{row['userId']},{row['movieId']},{row['rating']}\n")
            print(f"Predictions saved to {args.output_file}")
        if (args.alg == "SGD"):
            print("Prediction mode activated for SGD.")
            if not os.path.exists(args.model_path):
                print("Model file does not exist. Please run training first.")
                return
            with open(args.model_path, "rb") as f:
                model_data = pickle.load(f)
            predictions = predict_sgd(args.input_file, model_data)
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, "w") as f:
                f.write("userId,movieId,rating\n")
                for row in predictions:
                    f.write(f"{row['userId']},{row['movieId']},{row['rating']}\n")
            print(f"Predictions saved to {args.output_file}")
        if (args.alg == "ALL"):
            print("Prediction mode activated for all.")
            if not os.path.exists(args.model_path):
                print("Model file does not exist. Please run training first.")
                return
            with open(args.model_path, "rb") as f:
                models_data = pickle.load(f)
            predictions = predict_all(args.input_file, models_data)
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, "w") as f:
                f.write("userId,movieId,rating_nmf,rating_svd,rating_svd2,rating_sgd\n")
                for row in predictions:
                    f.write(f"{row['userId']},{row['movieId']},{row['rating_nmf']},{row['rating_svd']},"
                            f"{row['rating_svd2']},{row['rating_sgd']}\n")
            print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
