import torch
from nicoboco.operators import Operator


class Add(Operator):
    arity = 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class Sub(Operator):
    arity = 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x - y


class Mul(Operator):
    arity = 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class Div(Operator):
    arity = 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x / y


class Pow(Operator):
    arity = 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x**y


class Log(Operator):
    arity = 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.log(x) / torch.log(y)


class Sqrt(Operator):
    arity = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x)


class Square(Operator):
    arity = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class Sin(Operator):
    arity = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class Cos(Operator):
    arity = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)
