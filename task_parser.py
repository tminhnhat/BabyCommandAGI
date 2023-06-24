from typing import Dict, List

class TaskParser:

    def test(self):
        test_input = """type: command
path: /app/
```bash
sudo apt-get update
sudo apt-get install git
git clone https://github.com/flutter/flutter.git
```

type: command
path: /app/
```bash
cd flutter
./bin/flutter doctor
export PATH="$PATH:`pwd`/bin"
```

type: command
path: /app/
```bash
cd /app
flutter create my_flutter_app
```

type: command
path: /app/
```bash
flutter channel beta
flutter upgrade
flutter config --enable-web
```

type: command
path: /app/
```bash
cd my_flutter_app
flutter run -d web-server --web-port=8080
```

type: write
path: /app/Dockerfile
```
FROM ubuntu:latest

RUN apt-get update && apt-get install -y git

COPY . /app

WORKDIR /app

RUN git clone https://github.com/flutter/flutter.git

ENV PATH="/app/flutter/bin:${PATH}"

RUN flutter doctor

RUN cd /app && flutter create my_flutter_app

RUN flutter channel beta && flutter upgrade && flutter config --enable-web

EXPOSE 8080

CMD ["flutter", "run", "-d", "web-server", "--web-port=8080"]
```

type: plan
```
Configure the container to expose port 8080 to the host machine and access the app from a browser outside the container.
```"""
        parsed_data = self.decode(test_input)
        print(parsed_data)
        print(self.encode(parsed_data))

    def decode(self, input_string: str) -> List[dict]:
        data = input_string.strip().split("type:")[1:]
        if data.count == 0:
            raise ValueError("No valid items found")
        parsed_data = []

        for item in data:
            item = "type:" + item.strip()
            type, path, content = self._split_data(item)
            dict = {}
            dict["type"] = type
            if path is not None:
                dict["path"] = path
            dict["content"] = content
            parsed_data.append(dict)

        if parsed_data.count == 0:
            raise ValueError("No valid items found")

        return parsed_data
    
    def _split_data(self, input_data):
        lines = input_data.split('\n')
        type_line, path_line = None, None
        content_lines = []
        record_content = False

        for line in lines:
            line = line.strip()
            if line.startswith("type:"):
                if type_line is not None:
                    raise ValueError("Multiple type lines found")
                type_line = line[5:].strip()
            elif line.startswith("path:"):
                if path_line is not None:
                    raise ValueError("Multiple path lines found")
                path_line = line[5:].strip()
            elif line.startswith("```"):
                record_content = not record_content
            elif record_content:
                content_lines.append(line)

        if type_line is None:
            raise ValueError("No type line found")
        if content_lines.count == 0:
            raise ValueError("No content found")

        content = "\n".join(content_lines)

        return type_line, path_line, content
    
    def encode(self, input_data: List[dict]) -> str:
        output = ""
        for item in input_data:
            output += f"type: {item['type']}\n"
            if 'path' in item:
                output += f"path: {item['path']}\n"
            output += "```\n"
            output += f"{item['content']}\n"
            output += "```\n"
        return output
    
    def close_open_backticks(self, string: str) -> str:
        if string.count('```') % 2 != 0:
            string += '```'
        return string