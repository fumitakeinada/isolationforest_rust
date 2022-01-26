
extern crate ndarray;
use ndarray::{Array,ArrayView1,Array2,ShapeError};

extern crate rand;
use rand::{thread_rng, Rng};
use rand::distributions::{Uniform};
use rand_distr::{Distribution};

extern crate statrs;


use std::{thread, panic};
//use std::time::Duration;
use std::sync::{Arc, Mutex};
//use std::rc::Rc;


// ノードデータ設定
// 再帰処理が入るので、enumで定義
// 途中は2分木要素、末端は葉要素として定義
#[derive(Debug)]
pub enum IsolationNode {
    // 2分木の枝の情報
    Decision {
        left : Box<IsolationNode>, // 境界よりも小さいノード
        right: Box<IsolationNode>, // 境界以上のノード
        split_att: usize, // 分割に選んだ変数の列番号
        split_val: f64, // 境界値
    },
    // 葉の情報
    // 一つになるか、深さで打ち切りになるかで葉になるので、その時のデータのサイズ（行数）を記録
    // 末端のデータ自体は必要ない
    Leaf {
        size: usize, // 末端として残ったデータサイズ（行数）
        //data: Option<Array2<f64>>, // データ
    }
}

// ノード(再帰呼び出し対応）
impl IsolationNode {
    // 枝の場合の処理
    fn new_decision(
        left:Box<IsolationNode>, 
        right:Box<IsolationNode>, 
        split_att:usize,
        split_val: f64
        ) -> Box<IsolationNode>{
        let node = IsolationNode::Decision {
            left: left,
            right: right,
            split_att: split_att,
            split_val: split_val,
        };
        Box::new(node)
    }

    // 葉（末端）の場合の処理
    fn new_leaf(
        size:usize, 
        //data:Option<Array2<f64>>
        ) -> Box<IsolationNode>{

        let node = IsolationNode::Leaf {
            size: size, // データ数
            //data: data, // データ
        };
        Box::new(node)
    }
}


// アイソレーションフォレストの木の処理
pub struct IsolationTree {
    height: u32, // 深さ
    height_limit: u32, // 深さの最大値
}

impl IsolationTree {

    fn new(height:u32, height_limit:u32) -> Self{
        IsolationTree {height : height, height_limit : height_limit}
    }

    // ndarrayを渡し、学習させる
    fn fit(& mut self, x:Array2<f64>) -> Result< Box<IsolationNode>, ShapeError>{
        
        // データが孤立するか、指定の深さになった場合、葉を戻す。
         if x.nrows() <= 1 || self.height >= self.height_limit {
            //let node = IsolationNode::new_leaf(x.nrows(), Some(x));
            let node = IsolationNode::new_leaf(x.nrows());            
            return Ok(node);
         }
        
        // 変数の列をランダムに抽出（2分割する変数を選ぶ）
        let mut rng = thread_rng();
        let split_att: usize = rng.gen_range(0..x.ncols());
        
        // 列を選択
        let col = x.column(split_att);
        let vec_col = col.t().to_vec();

        // Vecで型がf64の場合の最大値、最小値選択方法
        let max = vec_col.iter().fold(0.0/0.0, |m, &v: &f64| v.max(m));
        let min = vec_col.iter().fold(0.0/0.0, |m, &v: &f64| v.min(m));
 
        // 閾値を決定
        let mut rng = thread_rng();

        let split_val:f64 = if min == max {
            min
        }
        else {
            rng.gen_range(min..max)
        };
    
        // 行毎に対象の変数が閾値より小さいか大きいかで分割        
        let mut x_left:Array2<f64> = Array::zeros((0,x.ncols())); //　行数0で初期化
        let mut x_right:Array2<f64> = Array::zeros((0,x.ncols())); //　行数0で初期化
        
        // スレッド化検討もしくは高速な並べ替え検討
        for i in x.rows(){
            if i.to_vec()[split_att] < split_val {
                // 行追加
                x_left.push_row(i)?;
            }
            else {
                // 行追加
                x_right.push_row(i)?;
            }
        }

        // 枝を作成し、深さを1加える
        let mut left = IsolationTree::new(self.height + 1, self.height_limit);
        let mut right = IsolationTree::new(self.height + 1, self.height_limit);

        // 分岐条件を設定し、枝を戻す。
        // 要スレッド化
        let left_node = left.fit(x_left)?;
        let right_node = right.fit(x_right)?;
        let node = IsolationNode::new_decision(left_node, right_node, split_att, split_val);
        Ok(node)
    }
}



// アンサンブル学習用のツリー集合
pub struct IsolationTreeEnsembleThread {
    sample_size: usize,
    n_trees: usize,
    tree_set: Arc<Mutex<Vec<IsolationNode>>>,
}


impl IsolationTreeEnsembleThread {
    pub fn new(sample_size:usize, n_trees:usize) -> Self{
        IsolationTreeEnsembleThread {
            sample_size: sample_size, 
            n_trees:n_trees, 
            tree_set: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn c(sample_size:usize) -> f64{
        const EULER_GAMMA:f64 = 0.5772156649; // Euler-Mascheroni constant
    
        if sample_size > 1 {
            let sizef:f64 = sample_size as f64;
            
            return  (2. * ((sizef - 1.).ln() + EULER_GAMMA) ) - ((2. *(sizef -1.))/sizef);        
        }
        else { // sample_size = 1
            return 0.;
        }
    }

    // データを渡して長さを返す
    fn tree_path_length(node:Box<&IsolationNode>, x:ArrayView1<f64>)->(usize, usize){
        // ノードを枝か葉で分ける

        match *node {
            IsolationNode::Decision {left, right, split_att, split_val}=> {
                // 対象の変数を取り出し、閾値の大小で枝を振り分ける。
                let direction = if x.to_vec()[*split_att] < *split_val {
                    left
                }
                else {
                    right
                };

                let result = Self::tree_path_length(Box::new(direction), x);
                let length = result.0 + 1;
                let decision_size = result.1;
        
                return (length, decision_size);
            }
            /*
            IsolationNode::Leaf {size, data:_ } => {
                return (1, *size);
            }
            */
            IsolationNode::Leaf {size} => {
                let length = 1;
                let size = *size;
                return (length, size);
            }
        }
    }

    fn make_isotree(x:&Arc<Array2<f64>>, sample_size:usize, height_limit:u32) -> Result<IsolationNode, ShapeError>{
        // 指定の件数分をランダムにデータ抽出（重複あり）
        let mut rng = thread_rng();
        let data_range = Uniform::new_inclusive(0, x.nrows() - 1);
        let data_rows: Vec<usize> = data_range.sample_iter(&mut rng).take(sample_size).collect();
        let mut random_data:Array2<f64> = Array::zeros((0,x.ncols()));
        for j in data_rows.iter(){
            random_data.push_row(x.row(*j))?;
        }
        
        // 一つのIsolationTreeを作成
        let mut isotree = IsolationTree::new(0, height_limit);
        let data_node = isotree.fit(random_data)?;
        Ok(*data_node)
    }


    // 学習
    pub fn fit(& mut self, x:Array2<f64>) -> Result<(), ShapeError>{

        // self.sample_size==0の場合は、入力データのデータ数をサンプルサイズに変更する。
        if self.sample_size == 0{
            self.sample_size = x.nrows();
        }

        // 深さの上限の設定
        let height_limit:u32 = (self.sample_size as f64).log2().ceil() as u32;
        let sample_size = self.sample_size;

        // 指定数のIsolationTreeの作成
        let n_trees = self.n_trees;
        let x_data = Arc::new(x);
        let tree_set = &self.tree_set;

        // スレッド化
        for _ in 0..n_trees{
            let hdl = thread::spawn({
                let x_data = x_data.clone();
                let tree_set = tree_set.clone();
                    move || {
                        let mut tree_set = tree_set.lock().unwrap();
                        let data = Self::make_isotree(&x_data, sample_size, height_limit);
                        match data {
                            Ok(ret) => (*tree_set).push(ret),
                            Err(e) => panic!("ShapeError: {:?}",e),
                        };
                }
            });
            
            hdl.join().unwrap();
        }
        Ok(())
    }

    // 行毎の長さの平均を算出
    fn get_path_length_mean(&self, row:ArrayView1<f64>)-> f64{
        let mut path:Vec<f64>= Vec::new();
        let tree_set = self.tree_set.lock().unwrap();
        // 要スレッド化
        for tree in tree_set.iter() {
            let result = Self::tree_path_length(Box::new(tree), row);
            // 孤立する前に既定の深さに達した場合、調整
            let result_decision_size = result.1;
            let pathlength = (result.0 as f64) + Self::c(result_decision_size);
            path.push(pathlength);
        }
        // データごとの平均値の算出
        let mut sum: f64 = 0.0;
        for j in path.iter() {
            sum += *j as f64;
        }
        let path_mean:f64 = sum/(path.len() as f64);
        path_mean
    }

    // データ毎の長さの平均を算出
    fn path_length_mean(&self, x:Array2<f64>) -> Vec<f64>{
        // 各データの深さの平均の格納場所
        let mut paths_mean:Vec<f64> = Vec::with_capacity(x.nrows());

        // 各データをツリーに当てはめる
        // スレッド化検討
        for i in 0..x.nrows(){
            let path_mean:f64 = self.get_path_length_mean(x.row(i));
            paths_mean.insert(i,path_mean);
        }
        paths_mean
    }

    // 異常度の算出
    pub fn anomaly_score(&mut self, x:Array2<f64>)  -> Vec<f64> {
        // 各データの深さの平均を算出し、異常値スコアを算出
        let paths_mean = self.path_length_mean(x);
        let mut anomaly_scores:Vec<f64> = Vec::with_capacity(paths_mean.len());
        let sample_size = self.sample_size;
 
        for (i,l) in paths_mean.iter().enumerate() {
            let c_val = Self::c(sample_size);
            let score =  (2. as f64).powf((-1.* l)/(c_val as f64));
            anomaly_scores.insert(i, score);
        }

        anomaly_scores
    }

}
