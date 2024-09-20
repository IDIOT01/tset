from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import matplotlib.pyplot as plt

# 使用你的浏览器驱动程序路径
service = Service('E:\\browser\\Chrome\\chromedriver_win32\\chromedriver.exe')

# 创建 WebDriver 实例，传入Service对象
driver = webdriver.Chrome(service=service)

# 打开指定的网页
driver.get("https://baidu.com")

# 等待网页完全加载
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

# 注入 JavaScript 代码来监听点击事件，并记录点击坐标
driver.execute_script("""
    window.recordedClicks = [];

    // 监听左键点击事件
    document.addEventListener('click', function(event) {
        window.recordedClicks.push({x: event.clientX, y: event.clientY, button: 'left'});
        console.log('Left click recorded at:', event.clientX, event.clientY);
    });

    // 监听右键点击事件
    document.addEventListener('contextmenu', function(event) {
        window.recordedClicks.push({x: event.clientX, y: event.clientY, button: 'right'});
        console.log('Right click recorded at:', event.clientX, event.clientY);
        event.preventDefault();  // 阻止默认的右键菜单弹出
    });

    console.log('Click listeners added successfully.');
""")

# 暂停几秒，等待你点击页面并记录位置
print("请点击网页记录坐标...")
time.sleep(10)  # 你可以调整此时间

# 从浏览器获取记录的点击位置
click_positions = driver.execute_script("return window.recordedClicks;")

# 检查获取到的点击位置
print(f"记录的点击位置: {click_positions}")

# 模拟点击网页的某个位置 (根据记录的坐标，使用 JavaScript 点击)
def js_click_at_position(x, y):
    # 使用 JavaScript 触发点击事件
    driver.execute_script(f"""
        var evt = new MouseEvent('click', {{
            bubbles: true,
            cancelable: true,
            view: window,
            clientX: {x},
            clientY: {y}
        }});
        var element = document.elementFromPoint({x}, {y});
        if (element) {{
            element.dispatchEvent(evt);
        }}
    """)

# 保存模拟点击的坐标
simulated_clicks = []

# 根据记录的坐标重复点击操作（仅左键点击）
if click_positions:
    for position in click_positions:
        if position['button'] == 'left':  # 仅模拟左键点击
            js_click_at_position(position['x'], position['y'])
            simulated_clicks.append((position['x'], position['y']))  # 保存坐标
            time.sleep(1)  # 每次点击间隔1秒
else:
    print("未记录到任何点击位置.")

# 关闭浏览器
driver.quit()

# ----------------------
# 绘制点击位置的散点图
# ----------------------
if simulated_clicks:
    x_coords, y_coords = zip(*simulated_clicks)  # 解压坐标列表
    plt.scatter(x_coords, y_coords, color='blue')
    plt.title('Simulated Click Positions')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.gca().invert_yaxis()  # Invert Y axis to match browser coordinates
    plt.show()
else:
    print("没有点击位置来绘制图像.")
