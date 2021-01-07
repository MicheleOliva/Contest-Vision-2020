import csv
import matplotlib.pyplot as plt


TRAINING_LOG_CSV_PATH = './training_log.csv'

epoch_num = []
training_loss = []
val_loss = []
training_mae = []
val_mae = []
with open(TRAINING_LOG_CSV_PATH, 'r') as csv_file:
	training_data = csv.reader(csv_file)
	i = 0
	for epoch in training_data: 
		# salta la prima riga perché contiene gli header
		if i == 0:
			i+=1
			continue
		epoch_num.append(int(epoch[0])+1)
		training_loss.append(round(float(epoch[1]), 2))
		training_mae.append(round(float(epoch[2]), 2))
		val_loss.append(round(float(epoch[4]), 2))
		val_mae.append(round(float(epoch[5]), 2))

# print di debug
print(f'epochs: {epoch_num}\ntraining loss: {training_loss}\ntraining mae: {training_mae}\nvalidation loss: {val_loss}\nvalidation mae: {val_mae}')

lr_changed = [7,13,16] # epoche dopo le quali abbiamo abbassato il LR 
lr_points_style = 'ro'
y_max_loss = 35
y_max_mae = 5

# plot loss
plt.figure()
plt.plot(epoch_num, training_loss)
plt.plot(epoch_num, val_loss, 'g-')
# settiamo valori asse x come interi
plt.xticks(epoch_num)
# cambiare range asse y
plt.axis([0, None, 0, y_max_loss])
# etichette sugli assi
plt.xlabel('Training epochs')
plt.ylabel('MSE')
# legenda
plt.legend(['Training', 'Validation'])
# aggiungiamo punti in cui abbiamo cambiato il learning rate
plt.plot(lr_changed, [training_loss[i-1] for i in lr_changed], lr_points_style)
plt.plot(lr_changed, [val_loss[i-1] for i in lr_changed], lr_points_style)
# titolo grafico
plt.title('MSE curves')
#plt.show()

# plot mae
plt.figure()
plt.plot(epoch_num, training_mae)
plt.plot(epoch_num, val_mae, 'g-')
# settiamo valori asse x come interi
plt.xticks(epoch_num)
# cambiare range asse y
plt.axis([0, None, 0, y_max_mae])
# etichette sugli assi
plt.xlabel('Training epochs')
plt.ylabel('MAE')
# legenda
plt.legend(['Training', 'Validation'])
# aggiungiamo punti in cui abbiamo cambiato il learning rate
plt.plot(lr_changed, [training_mae[i-1] for i in lr_changed], lr_points_style)
plt.plot(lr_changed, [val_mae[i-1] for i in lr_changed], lr_points_style)
# titolo grafico
plt.title('MAE curves')
plt.show()