#include "csvstream.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <set>
#include <map>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

// EFFECTS: Return a set of unique whitespace delimited words
set<string> unique_words(const string &str) {
  istringstream source(str);
  set<string> words;
  string word;
  while (source >> word) {
    words.insert(word);
  }
  return words;
}


class Classifier {
  private:
    int total_posts = 0; // total number of posts in entire training set
    set<string> vocabulary; // number of unique words in entire training set
    map<string, int> numPosts_word; // number of posts given word
    map<string, int> numPosts_label; // number of posts with label
    map<string, map<string, int>> numPosts_label_word; // number of posts given word
    vector<pair<string, string>> trainingData; // store training data
    
    
  public:
    // train on a CSV file
    void train(const string &filename) {

      ifstream fin(filename);
      if (!fin.is_open()) {
        cout << "Error opening file: " << filename << endl;
        return;
      }

      csvstream csv(fin);
      map<string, string> row;
   
      // read csv file
      while (csv >> row) {
        total_posts++;
        string label = row["tag"];
        numPosts_label[label]++;

        string content = row["content"];
        set<string> words = unique_words(content);

        trainingData.push_back({label, content});

        // iterate through words
        for(const auto &word : words) {
          vocabulary.insert(word);
          numPosts_word[word]++;
          numPosts_label_word[label][word]++;
        }
      }
      
    }

    // returns total number of training posts.
    int getTotalPosts() const {
      return total_posts;
    }
    
    // returns vocabulary size.
    int getVocabularySize() const {
      return vocabulary.size();
    }
    
    
    // compute the log-likelihood ln P(w | C) for a given word and label.
    double calculate_log_likelihood(const string &label, const string &word) const {
      
      int count = 0;
      auto labelIt = numPosts_label_word.find(label);
      if (labelIt != numPosts_label_word.end()) {
          auto wordIt = labelIt->second.find(word);
          if (wordIt != labelIt->second.end()) {
              count = wordIt->second;
          }
      }

      double log_likelihood = 0.0;
      // word appears in posts with label C.
      if (count > 0) {
        double probability = static_cast<double>(count) / numPosts_label.at(label);
        log_likelihood = log(probability);
    } 
      // word does not appear in posts with label C.
      else {
        int globalCount = 0;
        auto globalIt = numPosts_word.find(word);
        if (globalIt != numPosts_word.end()) {
            globalCount = globalIt->second;
        }
        if (globalCount > 0) {
            double probability = static_cast<double>(globalCount) / total_posts;
            log_likelihood = log(probability);
        }
        // word is unseen in the training data.
        else {
            double probability = 1.0 / total_posts;
            log_likelihood = log(probability);
        }
      }
      return log_likelihood;
    }


    void print_training_data() {
      cout << "training data:" << endl;
      for (const auto &example : trainingData) {
        cout << "  label = " << example.first << ", content = " 
          << example.second << endl;
      }
      cout << "trained on " << getTotalPosts() << " examples" << endl;
      cout << "vocabulary size = " << vocabulary.size() << endl;
      cout << endl;
    }
  
    
    // compute and print classifier parameters (log-priors and log-likelihoods)
    void print_classifier_parameters() {

        cout << "classes:" << endl;
        for (const auto &entry : numPosts_label) {
          string label = entry.first;
          int count = entry.second;
          double log_prior = log(static_cast<double>(count) / total_posts);
          cout << "  " << label << ", " << count << " examples, log-prior = " 
            << log_prior << endl;
      }

        cout << "classifier parameters:" << endl;
        for (auto &entry : numPosts_label_word) {
          string label = entry.first;
          map<string, int> map = entry.second;
          for (auto &it : map) {
            string word = it.first;
            int count = it.second;
            double log_likelihood = calculate_log_likelihood(label, word);

            cout << "  " << label << ":" << word << ", count = " << count
              << ", log-likelihood = " << log_likelihood << endl;

          }
      }
    }

  // predict the label for a new post and return the best label and its score
  string predict(const string &content, double &best_score) const {
    // extract unique words from the new post
    set<string> words = unique_words(content);
    best_score = numeric_limits<double>::lowest();
    string best_label;
  
    // evaluate each label
    for (const auto &label_entry : numPosts_label) {
        const string &label = label_entry.first;
        int label_count = label_entry.second;
        
        // compute the log prior once for this label
        double score = log(static_cast<double>(label_count) / total_posts);
        
        // for each unique word in the new post, add the log-likelihood
        for (const auto &w : words) {
            // Calculate log-likelihood for this word given the label
            score += calculate_log_likelihood(label, w);
        }
        
        // track the best-scoring label
        if (score > best_score || (score == best_score && label < best_label)) {
            best_score = score;
            best_label = label;
        }
    }
    return best_label;
  }
};

int main(int argc, char* argv[]) {
  // set floating point precision
  cout.precision(3);
  
  // check that the number of arguments is either 2 or 3
  if (argc < 2 || argc > 3) {
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
    return 1;
  }
  
  // get the training file name
  string train_file = argv[1];
  
  // create and train the classifier
  Classifier classifier;
  classifier.train(train_file);
  
  if (argc == 2) {
    classifier.print_training_data();
    classifier.print_classifier_parameters();
  }
  
  // if a test file is provided, perform prediction
  if (argc == 3) {
    string test_file = argv[2];
    ifstream fin(test_file);
    if (!fin.is_open()) {
      cout << "Error opening file: " << test_file << endl;
      return 1;
    }

    cout << "trained on " << classifier.getTotalPosts() << " examples" << endl << endl;
    cout << "test data:" << endl;
    
    // create a CSV stream to read the test file
    csvstream csv(fin);
    map<string, string> row;
    int correct = 0, total = 0;
    
    // process each row in the test file
    while (csv >> row) {
      total++;
      string true_label = row["tag"];
      string content = row["content"];
      
      // predict the label for the post; 'score' will hold the log-probability
      double score = 0;
      string predicted_label = classifier.predict(content, score);
      
      if (predicted_label == true_label) {
        correct++;
      }
      
      cout << "  correct = " << true_label 
           << ", predicted = " << predicted_label 
           << ", log-probability score = " << score << endl;
      cout << "  content = " << content << endl << endl;
    }
    
    cout << "performance: " << correct << " / " << total 
         << " posts predicted correctly" << endl << endl;
  }
  
  return 0;
}