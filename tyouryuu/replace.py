import re
import csv
def convert_to_comma_separated(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 各行の先頭の空白を除去し、その後、数値を半角スペース6個で区切り、それぞれの間にコンマを追加して出力ファイルに書き込む
            cleaned_line = line.strip().replace('      ', ',')
            
            
            cleaned_line = cleaned_line.replace('     ', ',')
            
            
            cleaned_line = cleaned_line.replace('    ', ',')
            
            
            cleaned_line = cleaned_line.replace('   ', ',')
            # outfile.write(cleaned_line)
            
            cleaned_line = cleaned_line.replace('  ', ',')
            outfile.write(cleaned_line+'\n')

            # outfile.write('\n')

# 

for i in range(3, 13):
    input_file = f'tyouryuu/250m_csv/cos-toriage_tosan_250m_2016{i:02d}_asc.csv'
    output_file = f'tyouryuu/prepro_250m_csv/edited_{i:02d}.csv'
    # スペースが6つ連続している場合、それらをシングルクォートに置き換える
    convert_to_comma_separated(input_file, output_file)