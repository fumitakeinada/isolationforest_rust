extern crate ndarray;
use ndarray::{Array,Array2,ArrayView,ShapeError};


use std::fs::File;
use std::io::BufReader;

extern crate statrs;

use env_logger;
use log::{error, warn, info, debug};
use std::env;

extern crate csv;
use csv::Error;

mod isolation_forest;

fn main() -> Result<(),Error> {
    env::set_var("RUST_LOG", "debug");
    env_logger::init();
    
    let arr: Array2<f64> = ndarray::arr2(&[
        [1.,5.,3.4,1.],
        [4.,6.,6.,10.,],
        [7.,8.,9.,8.4],
        [8.9,10.1,4.2,3.3],
        [530.7,4000.1,890.,10000.],
        [2.3,59.,6.4,10.3],
    ]);
    
    let csv = "1.0,5.0,3.4,1.0
4.0,6.0,6.0,10.
7.0,8.0,9.0,8.4
8.9,10.1,4.2,3.3
530.7,4000.1,890.0,10000.0
2.3,59.0,6.4,10.3";
    
    // CSVファイルの読み込み
    let train_csv_file = "train.csv";
    let test_normal_csv_file = "test_normal.csv";
    let test_abnormal_csv_file = "test_abnormal.csv";
    let csv_rows:usize = 2;

    let arr_train = trans_csv_to_arr(train_csv_file.to_string(), csv_rows).unwrap();
    let mut isotreeens = isolation_forest::IsolationTreeEnsembleThread::new(0, 400);
    isotreeens.fit(arr_train);


    let arr_test_normal = trans_csv_to_arr(test_normal_csv_file.to_string(),csv_rows).unwrap();
    let anomaly_scores_normal = isotreeens.anomaly_score(arr_test_normal);
    println!("{:?}", anomaly_scores_normal);

    let arr_test_abnormal = trans_csv_to_arr(test_abnormal_csv_file.to_string(), csv_rows).unwrap();
    let anomaly_scores_abnormal = isotreeens.anomaly_score(arr_test_abnormal);
    println!("{:?}", anomaly_scores_abnormal);

    Ok(())
}

fn trans_csv_to_arr(csv_file:String, rows:usize) -> Result<Array2<f64>,Error> {

    let file = File::open(csv_file)?;
    let buf = BufReader::new(file);

    let mut arrcsv:Array2<f64> = Array::zeros((0, rows));
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false).from_reader(buf);

    //let mut iter = rdr.into_records();
    
    for record in rdr.records(){
        let record = record?;
        let rowvec:Vec<f64> = record.iter().map(|x| x.parse().unwrap()).collect();
        arrcsv.push_row(ArrayView::from(&rowvec));
    }
    debug!("{:?}",arrcsv);
    Ok(arrcsv)    

}

