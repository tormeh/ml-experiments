use smartcore::dataset::Dataset;
use csv::Reader;
use serde::{Deserialize};
use std::fs::File;

#[derive(Deserialize)]
struct TitanicCSVpassenger {
    #[serde(rename(deserialize = "PassengerId"))]
    passenger_id: i16,
    #[serde(rename(deserialize = "Survived"))]
    survived: i8,
    #[serde(rename(deserialize = "Pclass"))]
    passenger_class: i8,
    #[serde(rename(deserialize = "Name"))]
    name: String,
    #[serde(rename(deserialize = "Sex"))]
    sex: String,
    #[serde(rename(deserialize = "Age"))]
    age: Option<f32>,
    #[serde(rename(deserialize = "SibSp"))]
    siblings_and_spouses: i8,
    #[serde(rename(deserialize = "Parch"))]
    parents_and_children: i8,
    #[serde(rename(deserialize = "Ticket"))]
    ticket: String,
    #[serde(rename(deserialize = "Fare"))]
    fare: f32,
    #[serde(rename(deserialize = "Cabin"))]
    cabin: Option<String>,
    #[serde(rename(deserialize = "Embarked"))]
    embarked: String 
}

type TitanicDataset = Dataset<f32, f32>;

fn main() {
    // DenseMatrix wrapper around Vec
    use smartcore::linalg::naive::dense_matrix::DenseMatrix;
    // SVM
    use smartcore::svm::svc::{SVCParameters, SVC};
    // Model performance
    use smartcore::metrics::roc_auc_score;
    use smartcore::model_selection::train_test_split;
    // Load dataset
    let titanic_data = load_titanic_data(read_titanic_train_data());
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        titanic_data.num_samples,
        titanic_data.num_features,
        &titanic_data.data,
    );
    // These are our target class labels
    let y = titanic_data.target;
    // Split dataset into training/test (80%/20%)
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    // SVC
    let y_hat_svm = SVC::fit(&x_train, &y_train, SVCParameters::default().with_c(10.0))
        .and_then(|svm| svm.predict(&x_test))
        .unwrap();
    // Calculate test error
    println!("AUC SVM: {}", roc_auc_score(&y_test, &y_hat_svm));
}

fn load_titanic_data(data: Vec<TitanicCSVpassenger>) -> TitanicDataset {
    let mut features = Vec::new();
    let mut survivals = Vec::new();
    let num_samples = data.len();
    for passenger in data {
        survivals.push(passenger.survived as f32);
        let sex: f32 = if passenger.sex.eq("male") {
            -1.0
        } else {
            1.0
        };
        features.push(sex);
    }
    let feature_names = vec!("sex".to_owned());
    let num_features = features.len()/num_samples;
    TitanicDataset {
        data: features,
        target: survivals,
        num_samples,
        num_features,
        feature_names,
        target_names: vec!("survived".to_owned()),
        description: "Titanic dataset".to_owned()
    }
}

fn read_titanic_train_data() -> Vec<TitanicCSVpassenger> {
    let mut vector = Vec::new();
    let file = File::open("../data/kaggle/titanic/train.csv").expect("no file");
    let mut reader = Reader::from_reader(file);
    for passenger_result in reader.deserialize::<TitanicCSVpassenger>() {

        vector.push(passenger_result.expect("bad passenger record"));
    }
    vector
}
