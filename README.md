# PICABOO - Graduate Team Project 1組 ＜韓組＞

![스크린샷 2023-06-23 오전 9 18 55](https://github.com/yeungjin-picaboo/picaboo-fastapi-server/assets/102615401/9d308608-bce5-4321-b199-7c27e441423a)

## プロジェクト情報
- サービス紹介
  - ユーザーが書いた日記の内容を絵に変換するサービス

- 開発期間・人数
  - 2023.02 ~ 2023.05 / 5名

## チームメンバー 🔥
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/KooAme" style="font-size: 5px;">
        <img src="https://github.com/KooAme.png" alt="member1" style="width: 80px;">
        <br>
        キム・ドング
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/woohyeon-dev">
        <img src="https://github.com/woohyeon-dev.png" alt="member2" width="80px">
        <br>
        クォン・ウヒョン
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/liosy1114" >
        <img src="https://github.com/liosy1114.png" alt="member3" width="80px">
        <br>
        キム・ジョンミン
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/wabwhc">
        <img src="https://github.com/wabwhc.png" alt="member4" width="80px">
        <br>
        パク・ムンシク
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/inhyoe">
        <img src="https://github.com/inhyoe.png" alt="member5" width="80px">
        <br>
        ユ・チャンフン
      </a>
    </td>
  </tr>
</table>
<br>

## スタートガイド
### 要求事項
- Node.js 18.16.1LTS
- Npm 9.6.6
- Python 3.10.11
- Android Studio 2022.1.1 Electric Eel
- PostgreSQL 13

### Installation
```ter
$ git clone https://github.com/yeungjin-picaboo
$ cd yeungjin-picaboo
```
### Frontend
```ter
$ cd picaboo-next-web
$ npm install
$ npm run build
$ npm run start
```
### Backend(NestJS)
```ter
$ cd picaboo-nest-server
$ npm install
$ npm run dev
```
### Backend(FastAPI)
```ter
$ cd picaboo-fastapi-server
$ pip install fastapi
$ pip install "uvicorn[standard]"
$ uvicorn main:app --reload --host=0.0.0.0 --port=9000
```
### Moblie(Kotlin) Use Android Studio
```ter
$ git clone https://github.com/yeungjin-picaboo/picaboo-kotlin-android
# Emulator execution
```
## 技術スタック 🔨
### Environment
<img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white"> <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"> <img src="https://img.shields.io/badge/visualstudiocode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white">
### Config
<img src="https://img.shields.io/badge/npm-CB3837?style=for-the-badge&logo=npm&logoColor=white"> <img src="https://img.shields.io/badge/yarn-2C8EBB?style=for-the-badge&logo=yarn&logoColor=white">
### Development
<img src="https://img.shields.io/badge/nextjs-000000?style=for-the-badge&logo=nextdotjs&logoColor=white"> <img src="https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white"> <img src="https://img.shields.io/badge/nestjs-E0234E?style=for-the-badge&logo=nestjs&logoColor=white"> <img src="https://img.shields.io/badge/kotlin-7F52FF?style=for-the-badge&logo=kotlin&logoColor=white"> <img src="https://img.shields.io/badge/solidity-363636?style=for-the-badge&logo=solidity&logoColor=white"> <img src="https://img.shields.io/badge/postgresql-4169E1?style=for-the-badge&logo=postgresql&logoColor=white"> <img src="https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white">
### Communication
<img src="https://img.shields.io/badge/slack-4A154B?style=for-the-badge&logo=slack&logoColor=white"> <img src="https://img.shields.io/badge/notion-000000?style=for-the-badge&logo=notion&logoColor=white"> <img src="https://img.shields.io/badge/figma-F24E1E?style=for-the-badge&logo=figma&logoColor=white">

## サービスの特徴
- ユーザーが書いた日記の内容を要約し、感情を分析する
- 要約した内容と感情をもとにAIが絵を生成してくれる
- 作成した絵に著作権を所有するためにNFT発行をすることができ、 マーケットを利用して販売も可能

## サービスの重要機能
- 日記作成、照会、修正、削除
- 日記の内容要約
- 作成したユーザーの位置に該当する天気及び作成した日記の感情分析
- 日記からキーワードを抽出して画像を生成
- NFT登録及び取引
- リストソート機能
  
