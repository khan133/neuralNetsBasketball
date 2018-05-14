import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.regularizers import WeightRegularizer, ActivityRegularizer, l2, activity_l2
from keras.optimizers import Adam
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt 


class BasketballPicks():
	def __init__(self):
		folder = '../data/'
		self.readFiles = []
		self.datasets = ["RegularSeasonDetailedResults.csv", "Seasons.csv", "Teams.csv", "TourneyDetailedResults.csv", "TourneySeeds.csv", "TourneySlots.csv"]
		for eachFile in self.datasets:
			self.readFiles.append(pd.read_csv(folder + eachFile))
		
		self.seasonResults = self.readFiles[0]
		self.teams = self.readFiles[2]
		self.tourney_results = self.readFiles[3]
		self.tourneySeeds = self.readFiles[4]
		self.trainYears = [2013]
		self.testYears = [2010]
	
	def startProcess(self):
		stats = self.getStats(self.trainYears)
		statsTest = self.getStats(self.testYears)
		#print statsTest	
		table, labels = self.label(stats, self.trainYears)
		table_test, labels_test = self.label(statsTest, self.testYears)
		#print table	
		X = normalize(table)
		y = labels
		Xt = normalize(table_test)
		yt = labels_test
		y2 = to_categorical(y)
		yt2 = to_categorical(yt)

		model = self.define_model(X)
		history = model.fit(X, y2, nb_epoch=100, batch_size=256)
		

		plotList = []
		newHistory = []
		for element in history.history['loss']:
			element /= 100
			newHistory.append(element)
		for i in range(1, 101):
			plotList.append(i)
		plt.plot(plotList, newHistory)
		plt.xlabel("Epoch Number")
		plt.ylabel("Loss")
		plt.show()

		print (history.history['loss'])

		plotList = []
		newHistory = []
		for element in history.history['acc']:
			element /= 1
			newHistory.append(element)
		for i in range(1, 101):
			plotList.append(i)
		plt.plot(plotList, newHistory)
		plt.xlabel("Epoch Number")
		plt.ylabel("Accuracy")
		plt.show()

		ys = model.predict_classes(Xt, batch_size=256)
		print("Neural Network Accuracy:")

		correct = 0
		for i in range(len(yt)):
			if ys[i] == yt[i]:
				correct += 1
		accuracy = float(correct) / len(yt)
		print(str(accuracy * 100) + "%")
	
	def getStats(self, years):
		rows = []
		for year in years: #go through all training years
			#print year
			row = []
			currYearResults = self.seasonResults[self.seasonResults['Season'] == year]
			#print currYearResults
			year_seeds = self.tourneySeeds[self.tourneySeeds["Season"] == year] 
		
			for team in year_seeds["Team"]:
		
				year_team = self.teams[self.teams["Team_Id"] == team]
			
				year_team = year_team.set_index("Team_Id").join(year_seeds.set_index("Team"))
				teamID = year_team.index.values[0]
				s = year_team["Seed"].values[0]
				year_team["Seed"] = int(s[1:3])

				gamesWon = currYearResults[currYearResults["Wteam"] == teamID]
				if len(gamesWon) == 0:
					continue
				gamesLost = currYearResults[currYearResults["Lteam"] == teamID]
				#print "Win games:"
				#print "Lossss:"
				year_games = pd.concat([gamesWon, gamesLost])
				dayNumWon = gamesWon["Daynum"]
				dn_loss = gamesLost["Daynum"]
				sum_win = dayNumWon.sum()
				sum_loss = dn_loss.sum()
				
				if sum_loss == 0:
					sum_loss = 0.00001
				wins = gamesWon.shape[0]

				losses = gamesLost.shape[0]
				played = wins + losses

				year_team["Wins"] = wins
				year_team["Losses"] = losses
				opponents = []
				for i, game in year_games.iterrows():
					#print "GOES IN HERE"
					if game["Wteam"] == team:
						opp = game["Lteam"]
					else:
						opp = game["Wteam"]
					opponents.append(opp)
				#print "LALLALALA"
				#print opponents
				opp_weighted_wins = []
				opp_weighted_losses = []
				for opp in opponents:
					opponentWon = currYearResults[currYearResults["Wteam"] == opp]
					opponentLoss = currYearResults[currYearResults["Lteam"] == opp]
					oppHome = opponentWon[opponentWon["Wloc"] == "H"].shape[0]
					oppLossHome = opponentLoss[opponentLoss["Wloc"] == "A"].shape[0]
					opp_a_w = opponentWon[opponentWon["Wloc"] == "A"].shape[0]
					opp_a_l = opponentLoss[opponentLoss["Wloc"] == "H"].shape[0]
					opp_n_w = opponentWon[opponentWon["Wloc"] == "N"].shape[0]
					opp_n_l = opponentLoss[opponentLoss["Wloc"] == "N"].shape[0]

					opp_weighted_wins.append((0.5 * oppHome) + opp_n_w + (1.2 * opp_a_w))
					opp_weighted_losses.append((0.4 * opp_a_l) + opp_n_l + (1.2 * opp_n_l))

				opp_ww = sum(opp_weighted_wins) / float(len(opp_weighted_wins))
				opp_wl = sum(opp_weighted_losses) / float(len(opp_weighted_losses))
				owp = opp_ww / (opp_ww + opp_wl)

				win_ftm = (gamesWon["Wftm"] * dayNumWon).sum() / sum_win
				loss_ftm = (gamesLost["Lftm"] * dn_loss).sum() / sum_loss
				win_fta = (gamesWon["Wfta"] * dayNumWon).sum() / sum_win
				loss_fta = (gamesLost["Lfta"] * dn_loss).sum() / sum_loss
				ftm = win_ftm + loss_ftm
				fta = win_fta + loss_fta
				year_team["FT%"] = float(ftm) / fta

				win_or = (gamesWon["Wor"] * dayNumWon).sum() / sum_win
				loss_or = (gamesLost["Lor"] * dn_loss).sum() / sum_loss
				win_dra = (gamesWon["Ldr"] * dayNumWon).sum() / sum_win
				loss_dra = (gamesLost["Wdr"] * dn_loss).sum() / sum_loss
				or_for = win_or + loss_or
				dra = win_dra + loss_dra
				offensive_glass = float(or_for) / (or_for + dra)
				year_team["Offensive Rebounding"] = offensive_glass

				win_dr = (gamesWon["Wdr"] * dayNumWon).sum() / sum_win
				loss_dr = (gamesLost["Ldr"] * dn_loss).sum() / sum_loss
				win_ora = (gamesWon["Lor"] * dayNumWon).sum() / sum_win
				loss_ora = (gamesLost["Wor"] * dn_loss).sum() / sum_loss
				dr_for = win_dr + loss_dr
				ora = win_ora + loss_ora
				defensive_glass = float(dr_for) / (dr_for + ora)
				year_team["Defensive Rebounding"] = defensive_glass

				year_team["Unique"] = year_team["Team_Name"] + str(year)

				row.append(year_team)
			row = pd.concat(row)
			rows.append(row)
		rows = pd.concat(rows)
		return rows

	def label(self, stats, years):
		table = []
		labels = []
		for year in years:
			result_year = self.tourney_results[self.tourney_results["Season"] == year]
			stats_year = stats[stats["Season"] == year]
			teams = result_year[["Wteam", "Lteam"]]
			for i, game in teams.iterrows():
				w = game["Wteam"]
				l = game["Lteam"]
				#if w not in stats_year["Team_ID"] or l not in stats_year["Team_ID"]:
				#	continue
				w_team = stats_year.loc[w]
				l_team = stats_year.loc[l]
				w_team = w_team.to_frame()
				l_team = l_team.to_frame()
				w_team = w_team.transpose()
				l_team = l_team.transpose()
				w_team = w_team.values
				l_team = l_team.values

				w_team = np.delete(w_team, -1)
				w_team = np.delete(w_team, 0)
				l_team = np.delete(l_team, -1)
				l_team = np.delete(l_team, 0)
				w_team = np.delete(w_team, 0)
				l_team = np.delete(l_team, 0)
				forward = np.append(w_team, l_team)
				backward = np.append(l_team, w_team)
				table.append(forward)
				table.append(backward)

				labels.append(1)
				labels.append(0)
		return table, labels

	def define_model(self, input):
		print len(input)
		model = Sequential([
			Dropout(0.1, input_shape=(len(input[0]),)),
			Dense(140, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01),),
			LeakyReLU(alpha=0.2),
			Dense(32),
			LeakyReLU(alpha=0.2),
			Dense(32),
			LeakyReLU(alpha=0.2),
			Dense(32),
			LeakyReLU(alpha=0.2),
			Dense(32),
			LeakyReLU(alpha=0.2),
			Dense(32),
			LeakyReLU(alpha=0.2),
			Dense(32),
			LeakyReLU(alpha=0.2),
			Dense(2),
			Activation("softmax"),
		])

		adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

		model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

		return model

				
X = BasketballPicks()
X.startProcess()		

