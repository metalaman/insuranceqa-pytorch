import pickle
import random
import numpy as np
import torch
import torch.autograd as autograd
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AnswerSelection(nn.Module):
    def __init__(self, conf):
        super(AnswerSelection, self).__init__()
        self.vocab_size = conf['vocab_size']
        self.hidden_dim = conf['hidden_dim']
        self.embedding_dim = conf['embedding_dim']
        self.question_len = conf['question_len']
        self.answer_len = conf['answer_len']
        self.batch_size = conf['batch_size']
        self.gpu_flag = conf['gpu']

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.cnns = nn.ModuleList([nn.Conv1d(self.hidden_dim, 500, filter_size, stride=1, padding=filter_size-(i+1)) for i, filter_size in enumerate([1,3,5])])
        self.question_maxpool = nn.MaxPool1d(self.question_len, stride=1)
        self.answer_maxpool = nn.MaxPool1d(self.answer_len, stride=1)
        self.dropout = nn.Dropout(p=0.2)
	self.init_weights()
	self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_len):
        if self.gpu_flag:
            return (autograd.Variable(torch.randn(2, batch_len, self.hidden_dim // 2)).cuda(),
                autograd.Variable(torch.randn(2, batch_len, self.hidden_dim // 2)).cuda())
        else:
            return (autograd.Variable(torch.randn(2, batch_len, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, batch_len, self.hidden_dim // 2)))

    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, question, answer):
        question_embedding = self.word_embeddings(question)
        answer_embedding = self.word_embeddings(answer)
        q_lstm, self.hidden = self.lstm(question_embedding, self.hidden)
        a_lstm, self.hidden = self.lstm(answer_embedding, self.hidden)
	q_lstm = q_lstm.contiguous()
	a_lstm = a_lstm.contiguous()
        q_lstm = q_lstm.view(-1,self.hidden_dim, self.question_len)
        a_lstm = a_lstm.view(-1,self.hidden_dim, self.answer_len)

        question_pool = []
        answer_pool = []
        for cnn in self.cnns:
            question_conv = cnn(q_lstm)
            answer_conv = cnn(a_lstm)
            question_max_pool = self.question_maxpool(question_conv)
            answer_max_pool = self.answer_maxpool(answer_conv)
            question_activation = F.tanh(torch.squeeze(question_max_pool))
            answer_activation = F.tanh(torch.squeeze(answer_max_pool))
            question_pool.append(question_activation)
            answer_pool.append(answer_activation)

        question_output = torch.cat(question_pool, dim=1)
        answer_output = torch.cat(answer_pool, dim=1)

        question_output = self.dropout(question_output)
        answer_output = self.dropout(answer_output)

        similarity = F.cosine_similarity(question_output, answer_output, dim=1)

        return similarity

    def fit(self, questions, good_answers, bad_answers):

        good_similarity = self.forward(questions, good_answers)
        bad_similarity = self.forward(questions, bad_answers)

        if self.gpu_flag:
        	zeros = autograd.Variable(torch.zeros(good_similarity.size()[0]), requires_grad=False).cuda()
        	margin = autograd.Variable(torch.linspace(0.05,0.05,good_similarity.size()[0]), requires_grad=False).cuda()
        else:
            zeros = autograd.Variable(torch.zeros(good_similarity.size()[0]), requires_grad=False)
            margin = autograd.Variable(torch.linspace(0.05,0.05,good_similarity.size()[0]), requires_grad=False)

    	loss = torch.max(zeros, autograd.Variable.sub(margin, autograd.Variable.sub(bad_similarity, good_similarity)))
        #similarity = torch.stack([good_similarity,bad_similarity],dim=1)
        #loss = torch.squeeze(torch.stack(map(lambda x: F.relu(0.05 - x[0] + x[1]), similarity), dim=0))
        accuracy = torch.eq(loss,zeros).type(torch.DoubleTensor).mean()
        return loss.sum(), accuracy.data[0]

class Evaluate():
    def __init__(self, conf):
        self.conf = conf
        self.all_answers = self.load('answers')
        self.vocab = self.load('vocabulary')
        self.conf['vocab_size'] = len(self.vocab) + 1
        self.model = AnswerSelection(self.conf)
        if conf['gpu']:
	           self.model.cuda()

    def load(self, name):
        return pickle.load(open('insurance_qa_python/'+name))

    def pad_question(self, data):
        return self.pad(data, self.conf.get('question_len', None))

    def pad_answer(self, data):
        return self.pad(data, self.conf.get('answer_len', None))

    def id_to_word(self, sentence):
        return [self.vocab.get(i,'<PAD>') for i in sentence]

    def pad(self, data, max_length):
        for i, item in enumerate(data):
            if len(item) >= max_length:
                data[i] = item[:max_length]
            elif len(item) < max_length:
                data[i] += [0] * (max_length - len(item))
        return data

    def train(self):
        batch_size = self.conf['batch_size']
        epochs = self.conf['epochs']
        training_set = self.load('train')

        questions = list()
        good_answers = list()
        for i, q in enumerate(training_set):
            questions += [q['question']] * len(q['answers'])
            good_answers += [self.all_answers[j] for j in q['answers']]

        questions = torch.LongTensor(self.pad_question(questions))
        good_answers = torch.LongTensor(self.pad_answer(good_answers))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf['learning_rate'])

        for i in xrange(epochs):
            bad_answers = torch.LongTensor(self.pad_answer(random.sample(self.all_answers.values(), len(good_answers))))
            train_loader = data.DataLoader(dataset=torch.cat([questions,good_answers,bad_answers],dim=1), batch_size=batch_size)
	    avg_loss = []
	    avg_acc = []
	    self.model.train()
            for step, train in enumerate(train_loader):
                if self.conf['gpu']:
                    batch_question = autograd.Variable(train[:,:self.conf['question_len']]).cuda()
                    batch_good_answer = autograd.Variable(train[:,self.conf['question_len']:self.conf['question_len']+self.conf['answer_len']]).cuda()
                    batch_bad_answer = autograd.Variable(train[:,self.conf['question_len']+self.conf['answer_len']:]).cuda()
                else:
                    batch_question = autograd.Variable(train[:,:self.conf['question_len']])
                    batch_good_answer = autograd.Variable(train[:,self.conf['question_len']:self.conf['question_len']+self.conf['answer_len']])
                    batch_bad_answer = autograd.Variable(train[:,self.conf['question_len']+self.conf['answer_len']:])
                optimizer.zero_grad()
		self.model.hidden = self.model.init_hidden(len(train))
		loss, acc = self.model.fit(batch_question, batch_good_answer, batch_bad_answer)
		avg_loss.append(loss.data[0])
		avg_acc.append(acc)
                loss.backward()
	        torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.25)
                optimizer.step()

	    print "Epoch: {0} Epoch Average loss: {1} Accuracy {2}".format(str(i), str(np.mean(avg_loss)), str(np.mean(avg_acc)))
            torch.save(self.model.state_dict(), "saved_model/answer_selection_model")

    def get_eval_sets(self):
        return dict([(s, self.load(s)) for s in ['dev', 'test1', 'test2']])

    def evaluate(self):
        #self.model.load_state_dict(torch.load("saved_model/answer_selection_model"))
        self.model = torch.load("saved_model/answer_selection_model")
        self.model.eval()
        eval_datasets = self.get_eval_sets()
        for name, dataset in eval_datasets.iteritems():
            print "Now evaluating : " + name
            for i, d in enumerate(dataset):
                indices = d['good'] + d['bad']
                answers = self.pad_answer([self.all_answers[i] for i in indices])
                question = self.pad_question([d['question']]*len(indices))
		print question.size(), answers.size()
                similarity = self.model.forward(question,answers)
                print similarity.size()
                break
            break

conf = {
    'question_len':20,
    'answer_len':150,
    'batch_size':256,
    'epochs':10000,
    'embedding_dim':256,
    'hidden_dim':256,
    'learning_rate':0.005,
    'margin':0.05,
    'gpu':1
}
ev = Evaluate(conf)
#ev.train()
ev.evaluate()
