import cv2
import numpy as np
import os
from tkinter import filedialog, messagebox
import tkinter as tk

def check_image_file(image_path):
    """
    이미지 파일의 유효성을 검사하는 함수
    
    Parameters:
    - image_path: 이미지 파일 경로
    
    Returns:
    - (성공 여부, 오류 메시지)
    """
    # 파일 존재 여부 확인
    if not os.path.exists(image_path):
        return False, "파일이 존재하지 않습니다."
    
    # 파일 읽기 권한 확인
    if not os.access(image_path, os.R_OK):
        return False, "파일에 대한 읽기 권한이 없습니다."
    
    # 파일 크기 확인
    if os.path.getsize(image_path) == 0:
        return False, "파일이 비어있습니다."
    
    # 파일 확장자 확인
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in valid_extensions:
        return False, f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(valid_extensions)}"
    
    return True, ""

def adjust_brightness(image, brightness_factor):
    """
    이미지의 밝기를 조절하는 함수
    
    Parameters:
    - image: 입력 이미지
    - brightness_factor: 밝기 조절 계수 (1.0이 원본, 1.0보다 크면 밝기 증가, 작으면 밝기 감소)
    
    Returns:
    - 밝기가 조절된 이미지
    """
    # 밝기 조절
    result = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return result

def adjust_saturation(image_path, saturation_factor):
    """
    이미지의 채도를 조절하는 함수
    
    Parameters:
    - image_path: 이미지 파일 경로
    - saturation_factor: 채도 조절 계수 (1.0이 원본, 1.0보다 크면 채도 증가, 작으면 채도 감소)
    
    Returns:
    - 채도가 조절된 이미지
    """
    # 이미지 파일 유효성 검사
    is_valid, error_message = check_image_file(image_path)
    if not is_valid:
        print(f"Error: {error_message}")
        messagebox.showerror("오류", error_message)
        return None
    
    # 이미지 읽기
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: 이미지를 읽을 수 없습니다. 파일이 손상되었을 수 있습니다.")
            messagebox.showerror("오류", "이미지를 읽을 수 없습니다. 파일이 손상되었을 수 있습니다.")
            return None
        
        # 이미지 크기 확인
        if image.size == 0:
            print(f"Error: 이미지 크기가 0입니다.")
            messagebox.showerror("오류", "이미지 크기가 0입니다.")
            return None
            
        # BGR에서 HSV로 변환
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 채도 채널에 saturation_factor 적용
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)
        
        # HSV에서 BGR로 다시 변환
        result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        return result_image
        
    except Exception as e:
        print(f"Error: 이미지 처리 중 오류가 발생했습니다: {str(e)}")
        messagebox.showerror("오류", f"이미지 처리 중 오류가 발생했습니다: {str(e)}")
        return None

def save_image(image, output_path):
    """
    이미지를 저장하는 함수
    
    Parameters:
    - image: 저장할 이미지
    - output_path: 저장 경로
    
    Returns:
    - 성공 여부
    """
    try:
        # 디렉토리가 존재하는지 확인
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 이미지 저장
        success = cv2.imwrite(output_path, image)
        
        if not success:
            print(f"Error: Failed to save image to {output_path}")
            return False
            
        return True
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False

def main():
    # Tkinter 윈도우 생성 (파일 선택 다이얼로그용)
    root = tk.Tk()
    root.withdraw()  # 메인 윈도우 숨기기
    
    # 이미지 파일 선택
    print("처리할 이미지 파일을 선택해주세요...")
    input_image = filedialog.askopenfilename(
        title="이미지 파일 선택",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not input_image:
        print("이미지가 선택되지 않았습니다. 프로그램을 종료합니다.")
        return
    
    # 결과 저장 폴더 선택
    print("결과 이미지를 저장할 폴더를 선택해주세요...")
    output_dir = filedialog.askdirectory(title="결과 저장 폴더 선택")
    
    if not output_dir:
        print("저장 폴더가 선택되지 않았습니다. 프로그램을 종료합니다.")
        return
    
    # 저장 폴더에 쓰기 권한 확인
    if not os.access(output_dir, os.W_OK):
        print(f"Error: No write permission for directory: {output_dir}")
        messagebox.showerror("Error", f"선택한 폴더에 쓰기 권한이 없습니다: {output_dir}")
        return
    
    # 이미지 읽기
    image = cv2.imread(input_image)
    if image is None:
        print("Error: 이미지를 읽을 수 없습니다.")
        return
    
    # 밝기 조절 계수 (1.0이 원본, 1.0보다 크면 밝기 증가, 작으면 밝기 감소)
    brightness_factors = [0.5, 1.0, 1.5]  # 밝기 감소, 원본, 밝기 증가
    
    # 각각의 밝기 계수에 대해 이미지 처리
    for factor in brightness_factors:
        # 밝기 조절
        result = adjust_brightness(image, factor)
        
        # 결과 이미지 저장
        filename = os.path.basename(input_image)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_brightness_{factor}{ext}")
        
        # 이미지 저장 시도
        if save_image(result, output_path):
            print(f"Successfully saved image with brightness factor {factor} to {output_path}")
        else:
            print(f"Failed to save image with brightness factor {factor}")
            continue
        
        # 결과 이미지 표시
        cv2.imshow(f"Brightness Factor: {factor}", result)
        print(f"Press any key to continue... (Factor: {factor})")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 채도 조절 계수 (1.0이 원본, 1.0보다 크면 채도 증가, 작으면 채도 감소)
    saturation_factors = [0.5, 1.0, 1.5]  # 채도 감소, 원본, 채도 증가
    
    # 각각의 채도 계수에 대해 이미지 처리
    for factor in saturation_factors:
        # 채도 조절
        result = adjust_saturation(input_image, factor)
        
        if result is not None:
            # 결과 이미지 저장
            filename = os.path.basename(input_image)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_saturation_{factor}{ext}")
            
            # 이미지 저장 시도
            if save_image(result, output_path):
                print(f"Successfully saved image with saturation factor {factor} to {output_path}")
            else:
                print(f"Failed to save image with saturation factor {factor}")
                continue
            
            # 결과 이미지 표시
            cv2.imshow(f"Saturation Factor: {factor}", result)
            print(f"Press any key to continue... (Factor: {factor})")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("\n모든 처리가 완료되었습니다.")
    print(f"결과 이미지는 '{output_dir}' 폴더에 저장되었습니다.")
    
    # 저장된 이미지 확인 메시지
    messagebox.showinfo("완료", f"모든 처리가 완료되었습니다.\n결과 이미지는 '{output_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 