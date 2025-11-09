import logging, os
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.mda import featImpMDA
from src.evaluation.kFold import cvScore, PurgedKFold
from src.evaluation.sfi import auxFeatImpSFI, map_featImpSfi
from src.evaluation.feat_pca import compare_imp, orthoFeats


def main(configs, dataset: pd.DataFrame, logger, run):
    logger.info(f'-'*50)
    logger.info(f"Training Pipeline started at: {dt.datetime.now().strftime('%Y%m%d%H%M')}")

    # Initialize classifier
    rf_clf = RandomForestClassifier()

    # Separate features, labels, and weights
    X = dataset.drop(['labels', 't1', 'weights', 'timestamp'], axis=1)
    y = dataset['labels']
    t1 = dataset['t1']
    weights = dataset['weights']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    wghts_train, wghts_test = weights.iloc[:len(X_train)], weights.iloc[len(X_train):]

    # Fit and evaluate classifier
    rf_clf.fit(X_train, y_train, sample_weight=wghts_train)
    preds = rf_clf.predict(X_test)
    score = accuracy_score(y_test, preds, sample_weight=wghts_test)
    logger.info(f"Model achieved an accuracy of: {score:.4f}")

    # Cross-validation score using PurgedKFold
    score = cvScore(
        rf_clf, X, y, weights, t1=t1,
        scoring=configs.scoring,
        pctEmbargo=configs.pctEmbargo,
        cv=configs.cv
    )
    logger.info(f"The score after {configs.cv} PurgedKFold CVs is: {score}")

    # confusion matrix plot
    path = f'runs/{run}/results'
    try:
        os.makedirs(path, exist_ok=True)

        c_matrix = confusion_matrix(
            y_true=y_test, y_pred=preds,
            sample_weight=wghts_test,
            labels=configs.labels
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            c_matrix, annot=True, fmt='g', cmap='Blues',
            cbar=True, xticklabels=configs.labels, yticklabels=configs.labels
        )
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        save_path = os.path.join(path, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved at: {save_path}")

    except Exception as e:
        logger.exception(f"Error during creation and saving of confusion matrix: {e}")
        raise

    # MDI Feature Importance
    if configs.model == 'tree':
        try:
            # Corrected the column drop (your original had a typo in the string)
            feature_names = dataset.columns.drop(['t1', 'labels', 'weights', 'timestamp'])
            fi = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_clf.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(8, max(4, 0.4 * len(fi))))
            sns.barplot(x='importance', y='feature', data=fi, hue='feature',
                         dodge=False, palette='Blues_r', legend=False)
            plt.title('Feature Importance (MDI)')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.tight_layout()

            save_path = os.path.join(path, "mdi.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            logger.info(f"MDI feature importance saved at: {save_path}")

        except Exception as e:
            logger.exception(f"Error during plotting/saving of MDI: {e}")

    # MDA Feature Importance
    try:
        imp, src = featImpMDA(
            clf=rf_clf, X=X, y=y, sample_weight=weights,
            scoring=configs.scoring, cv=configs.cv,
            t1=t1, pctEmbargo=configs.pctEmbargo
        )

        logger.info("Mean and std deviation by each feature in MDA:")
        logger.info(f"\n{imp}")
        logger.info(f"The MDA score is: {src:.4f}")

        imp = imp.sort_values(by='mean', ascending=True)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='mean', y=imp.index, data=imp, palette='Greens_r')
        plt.xlabel('Mean Decrease Accuracy')
        plt.ylabel('Features')
        plt.title('Feature Importance (MDA)')
        plt.tight_layout()

        save_path = os.path.join(path, 'mda.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"MDA feature importance saved at: {save_path}")

    except Exception as e:
        logger.exception(f"Error while calculating/plotting MDA feature importance: {e}")

    # single feature importance
    try:
        # determining sfi throught cvscore method
        imp = auxFeatImpSFI(featNames=feature_names, clf=rf_clf, trnsX=X,
                            y=y, sample_weight=weights, scoring=configs.scoring, 
                            pctEmbargo=configs.pctEmbargo, t1=t1, cv=configs.cv)
        
        # plotting and saving it in folder path
        save_path = map_featImpSfi(imp, folder_path=path)
        logger.info(f"SFI feature importanced saved at : {save_path}")

    except Exception as e:
        logger.exception(f"Error while calculating/plotting SFI feat imp: {e}")

    # pca unsupervised feature importance
    try:
        # determing the feature importance using eigenvalues
        dfp, eVal, eVecv = orthoFeats(dfX=X)

        mdi = pd.Series(
        rf_clf.feature_importances_,
        index=[f'PC_{i+1}' for i in range(len(rf_clf.feature_importances_))]
        )

        # comparing pca unsupervised feature importance with mdi feature importance
        save_path = compare_imp(eVal, mdi=mdi, save_path=path)
    except Exception as e:
        logger.exception(f"Error while calculating/comparing pca&mdi : {e}")



if __name__ == '__main__':
    from pipelines.dataPipeline import data_pipeline
    from config import ModelConfig, DataConfig
    
    mconfig, dconfig = ModelConfig(), DataConfig()
    dataset, logger, run = data_pipeline(dconfig)
    main(mconfig, dataset, logger, run)
    