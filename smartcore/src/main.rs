fn main() {
    use smartcore::dataset::*;
    // DenseMatrix wrapper around Vec
    use smartcore::linalg::naive::dense_matrix::DenseMatrix;
    // SVM
    use smartcore::svm::svc::{SVCParameters, SVC};
    // Model performance
    use smartcore::metrics::roc_auc_score;
    use smartcore::model_selection::train_test_split;
    // Load dataset
    let cancer_data = breast_cancer::load_dataset();
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        cancer_data.num_samples,
        cancer_data.num_features,
        &cancer_data.data,
    );
    // These are our target class labels
    let y = cancer_data.target;
    // Split dataset into training/test (80%/20%)
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    // SVC
    let y_hat_svm = SVC::fit(&x_train, &y_train, SVCParameters::default().with_c(10.0))
        .and_then(|svm| svm.predict(&x_test))
        .unwrap();
    // Calculate test error
    println!("AUC SVM: {}", roc_auc_score(&y_test, &y_hat_svm));
}
