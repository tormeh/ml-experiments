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
    display_dataset(&titanic_data);
    //display_dataset(&smartcore::dataset::breast_cancer::load_dataset());
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
    let y_hat_svm = SVC::fit(&x_train, &y_train, SVCParameters::default())
        .and_then(|svm| svm.predict(&x_test))
        .unwrap();
    // Calculate test error
    println!("AUC SVM: {}", roc_auc_score(&y_test, &y_hat_svm));
}

fn load_titanic_data(data: Vec<TitanicCSVpassenger>) -> TitanicDataset {
    let mut features = Vec::new();
    let mut survivals = Vec::new();
    let num_samples = data.len();
    let class_normalizer = get_normalizer(data.iter().map(|pass| pass.passenger_class as f32).collect());
    let par_ch_normalizer = get_normalizer(data.iter().map(|pass| pass.parents_and_children as f32).collect());
    let sib_sp_normalizer = get_normalizer(data.iter().map(|pass| pass.siblings_and_spouses as f32).collect());
    let fare_normalizer = get_normalizer(data.iter().map(|pass| pass.fare as f32).collect());
    let ages: Vec<f32> = data.iter().flat_map(|pass| pass.age).collect();
    let average_age: f32 = ages.iter().sum::<f32>()/(ages.len() as f32);
    let age_normalizer = get_normalizer(ages);
    for passenger in data {
        survivals.push(passenger.survived as f32);
        let sex: f32 = if passenger.sex.eq("male") {
            0.0
        } else {
            1.0
        };
        features.push(sex);
        features.push(class_normalizer(passenger.passenger_class as f32));
        features.push(par_ch_normalizer(passenger.parents_and_children as f32));
        features.push(sib_sp_normalizer(passenger.siblings_and_spouses as f32));
        features.push(fare_normalizer(passenger.fare as f32));
        features.push(age_normalizer(passenger.age.unwrap_or(average_age) as f32));
    }
    let feature_names = vec!(
        "sex".to_owned(),
        "passenger class".to_owned(),
        "parents_and_children".to_owned(),
        "siblings_and_spouses".to_owned(),
        "fare".to_owned(),
        "age".to_owned(),
    );
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

fn get_normalizer(data: Vec<f32>) -> Box<dyn Fn(f32) -> f32> {
    let sorted = {
        let mut copy = data.to_vec();
        copy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        copy
    };
    let min = sorted[0];
    let max = sorted[sorted.len()-1];
    let max_diff = max-min;
    Box::new(move |num: f32| (num-min)/max_diff)
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

fn display_dataset<X: Copy + std::fmt::Debug, Y: Copy + std::fmt::Debug>(dataset: &Dataset<X, Y>) {
    struct Target<Y> {
        name: String,
        value: Y
    }
    struct Feature<X> {
        name: String,
        value: X
    }
    struct DataPoint<X, Y> {
        labels: Vec<Target<Y>>,
        features: Vec<Feature<X>>
    }
    impl <X: Copy + std::fmt::Debug, Y: Copy + std::fmt::Debug>std::fmt::Display for DataPoint<X, Y> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            // Write strictly the first element into the supplied output
            // stream: `f`. Returns `fmt::Result` which indicates whether the
            // operation succeeded or failed. Note that `write!` uses syntax which
            // is very similar to `println!`.
            write!(
                f, "{} : {}",
                self.labels.iter().map(|target| format!("{}:{:?}, ", target.name, target.value)).collect::<String>(),
                self.features.iter().map(|feature| format!("{}:{:?}, ", feature.name, feature.value)).collect::<String>()
            )
        }
    }
    println!("{}", dataset.description);
    let mut datapoints = Vec::new();
    for sample_index in 0..dataset.num_samples {
        let mut features = Vec::new();
        for feature_index in 0..dataset.feature_names.len() {
            features.push(Feature{
                name: dataset.feature_names[feature_index].to_owned(),
                value: dataset.data[sample_index*dataset.num_features+feature_index]
            });
        }
        let mut targets = Vec::new();
        for target_index in 0..dataset.target_names.len() {
            targets.push(Target{
                name: dataset.target_names[target_index].to_owned(),
                value: dataset.target[sample_index*dataset.target_names.len()+target_index]
            });
        }
        datapoints.push(DataPoint {
            labels: targets,
            features
        })
    }
    for point in datapoints {
        println!("{}", point);
    }
}
