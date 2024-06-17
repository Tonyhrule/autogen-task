freader = open('inverse_matrix.in', 'r')
n = int(freader.readline())
matrix = []
for _ in range(n):
    matrix.append(list(map(int, freader.readline().split())))
freader.close()

identity_matrix = [[1,0,0],[0,1,0],[0,0,1]]
row_index = list(range(n))

#going through each row in the matrix
for row in range(n): #finding equalizer to make diagonals as 1
    equalizer = 1/matrix[row][row]

    for col in range(n): #getting 1s in the diagonal
        matrix[row][col] = matrix[row][col] * equalizer
        identity_matrix[row][col] = identity_matrix[row][col] * equalizer
    
    #making the outsides zeros
    for out_x in row_index[:row] + row_index[row + 1:]: #finding the equalizer to make zeros
        equalizer2 = matrix[out_x][row]
        
        for out_y in range(n): #applying the equalizer to make the outsides zeroes
            matrix[out_x][out_y] = matrix[out_x][out_y] - equalizer2 * matrix[row][out_y]
            identity_matrix[out_x][out_y] = identity_matrix[out_x][out_y] - equalizer2 * identity_matrix[row][out_y]

print(identity_matrix)
