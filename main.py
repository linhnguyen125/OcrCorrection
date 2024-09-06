import time

from tool.predictor import Predictor


if __name__ == '__main__':
    model_predictor = Predictor(device='cpu')
    unacc_paragraphs = [
        "Trên cơsở kếT quả kiểm tra hiện trạng, Tòa an nhân dân tối cao xẻm xét, lap phương án sắp xếp lại, xử lý \
        các cơ sở nhà, đất thuộc phám vi Quản lý, gửi lấy ý kiến của Ủy ban nhân dân cấp tỉnh nơi có nhà, đất. \
        Riêng đổi với việc tổ chức kiểm tra hiện trạng, lập phương án, phê duyệt phương án sắp xếp lại, xử lý nhà, \
        đất trên địa bàn các thành phố Hà Nội, Hồ Chí Minh, Đà Nẵng, Cần Thơ, Hải Phòng do Bộ Tài chính thực hiện theo \
        quy định tại Mục 3, Điều 5 Nghị định số 167/2017/NĐ-CP: ",
        
        "Căn cứ Quyếtđịnh số 676/2016/QĐ-TANDTC-KHTC ngày 13/9/2016 của Chánh án Tòa án nhân dân tối cao về \
        việc phân cấp quản lý ngân sách Nhà nước và quản lý dự án đầu tư xây dựng công trình trụ sở làm việc \
        Tòa án địa phương Để đảm bảo quản lý, sử dụng có hiệu quả các cơ sở nhà, đất trong hệ thống Tòa án nhân dân, \
        Tòa án nhân dân tối cao yêu cầu Thủ trưởng các đơn vị quán triệt, nghiêm túc thực hiện Luật Quản lý, sử dụng \
        tài sản công, các văn bản pháp luật có liên quan và hướng dan trình tự, thủ tục sắp xếp lại, xử lý nhà, đat \
        như sau: ",
        "ĐIỀU 6 : ĐIỀU KHOẢN VỀ BẢO DƯỜNG, SỬA CHỮA NHÀ & CÁC TRANG THIẾT BỊ"
    ]
    
    start = time.time()
    print('Results: ')
    for i, p in enumerate(unacc_paragraphs):
        outs = model_predictor.predict(p.strip(), NGRAM=5)
        print("==============================================================================================")
        print(p)
        print("----------------------------------------------------------------------------------------------")
        print(outs)
    end = time.time()