import argparse
import pandas as pd # type: ignore
from src.objective_extractor import ObjectiveExtractor

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--document', type=str, help='Document (raw text) to extract the objective from.', required=True)
    parser.add_argument('--token_starts', type=str, help='Token starts.', required=False, default=[0, 1000, 2000, 3000, 4000, 5000])
    #token_starts=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    args = parser.parse_args()

    df = pd.DataFrame([{"text": args.document}])
    
    extractor = ObjectiveExtractor(do_train=False, trained_promt="src/templates/ObjectiveExtractor-saved.json")
    df = extractor.predict(df, col_extract="text", token_starts=args.token_starts)

    print(df.to_dict(orient="records")[0])

    return


if __name__ == "__main__":
    main()